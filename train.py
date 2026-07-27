import argparse
from json import encoder
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import *
from utils.models import *
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True

def parse_arguments():
    parser = argparse.ArgumentParser(description="AdaIN Neural Style Transfer")

    parser.add_argument(
        "--content_dir",
        type=str,
        default="content_data",
        help="Directory containing content images."
    )

    parser.add_argument(
        "--style_dir",
        type=str,
        default="style_data",
        help="Directory containing style images."
    )

    parser.add_argument(
        "--vgg",
        type=str,
        default="vgg_normalised.pth",
        help="Path to pretrained VGG weights."
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiment1",
        help="Experiment folder name."
    )
    parser.add_argument(
        '--final_size',
        type=int,
        default=256,
        help='final image size'
    )
    parser.add_argument(
        '--content_size',
        type=int,
        default=512,
        help='content image size'
    )
    parser.add_argument(
        '--style_size',
        type=int,
        default=512,
        help='style image size'
    )
    parser.add_argument(
        '--crop',
        action='store_true',
        default = True,
        help='crop images'
    )
    parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Batch size for training."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=5e-5,
        help="Learning rate decay factor."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--content_weight",
        type=float,
        default=1.0,
        help="Weight for content loss."
    )
    parser.add_argument(
        "--style_weight",
        type=float,
        default=10.0,
        help="Weight for style loss."
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Interval (in epochs) for logging training progress."
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=2,
        help="Interval (in epochs) for saving model checkpoints."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a checkpoint."
    )
    parser.add_argument(
        "--decoder_path",
        type=str,
        default=None,
        help="Path to the decoder model checkpoint."
    )
    parser.add_argument(
        "--optimized_path",
        type=str,
        default=None,
        help="Path to the optimizer state checkpoint."
    )
    return parser.parse_args()


def check_paths(*paths):
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")


def save_arguments(args, save_dir):
    with open(save_dir / "args.txt", "w") as file:
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device : {device}")

    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)
    vgg_path = Path(args.vgg)

    check_paths(content_dir, style_dir, vgg_path)

    save_dir = Path("experiments") / args.experiment_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    save_arguments(args, save_dir)

    content_transform = get_transform(args.content_size,args.crop,args.final_size)  # Define your content image transformations here
    style_transform = get_transform(args.style_size,args.crop,args.final_size)  # Define your style image transformations here

    content_dataset = ImageFolderDataset(content_dir, transform=content_transform)
    style_dataset = ImageFolderDataset(style_dir, transform=style_transform)

    content_dataloader = DataLoader(
    content_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2,
    )

    style_dataloader = DataLoader(
    style_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2,
    )
    print(len(content_dataloader),len(style_dataloader))
    
    encoder = VGGEncoder(vgg_path).to(
    device,
    )
    decoder = Decoder().to(
    device,
    )
    encoder.requires_grad_(False)
    encoder.eval()
   
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda =lambda epoch :1.0/(1.0 +args.lr_decay*epoch))
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.decoder_path, map_location=device)

        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resuming from Epoch {start_epoch}")
    mse_loss = torch.nn.MSELoss()
    encoder.eval()  # Set encoder to evaluation mode
    running_loss = None
    running_closs = None
    running_sloss = None
    for epoch in range(start_epoch, args.epochs):
        progress_bar = tqdm(zip(content_dataloader , style_dataloader), total=min(len(content_dataloader), len(style_dataloader)), )
        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0
        for content_batch ,style_batch in progress_bar:
            content_batch = content_batch.to(device,non_blocking=True)
            style_batch = style_batch.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                c_feats = encoder(content_batch)
                s_feats = encoder(style_batch)

                t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

                g = decoder(t)
                g_feats = encoder(g)

                loss_c = mse_loss(g_feats[-1], t) * args.content_weight

                loss_s = 0
                for gf, sf in zip(g_feats, s_feats):
                    g_mean, g_std = calc_mean_std(gf)
                    s_mean, s_std = calc_mean_std(sf)

                    loss_s += mse_loss(g_mean, s_mean)
                    loss_s += mse_loss(g_std, s_std)

                loss_s *= args.style_weight
                loss = loss_c + loss_s

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_description(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {loss.item():.4f} Content Loss: {loss_c.item():.4f} Style Loss: {loss_s.item():.4f}")
            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
        scheduler.step()
        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)
        if (epoch +1) % args.log_interval == 0:
            tqdm.write(f"Iter {epoch + 1} Loss: {running_loss:.4f} Content Loss: {running_closs:.4f} Style Loss: {running_sloss:.4f}")
        if (epoch+1)%args.save_interval==0:
            checkpoint = {
                "epoch": epoch,
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }

            torch.save(
                checkpoint,
                save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            )
            print(
                f"Saved checkpoint at epoch {epoch + 1} to "
                f"{save_dir / f'checkpoint_epoch_{epoch + 1}.pth'}"
            )
            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f"output_epoch_{epoch + 1}.png", nrow=args.batch_size)

if __name__ == "__main__":
    main()
