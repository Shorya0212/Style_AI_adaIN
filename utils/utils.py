import os
import random
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()

        self.root = root
        self.transform = transform

        self.files = [
            f for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        while True:
            img_path = os.path.join(self.root, self.files[index])

            try:
                image = Image.open(img_path).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                return image

            except Exception as e:
                print(f"\nSkipping corrupted image: {img_path}")
                print(e)

                index = random.randint(0, len(self.files) - 1)


def get_transform(size, crop, final_size):
    from torchvision import transforms

    transform_list = []

    if size > 0:
        transform_list.append(transforms.Resize(size))

    if crop:
        transform_list.append(transforms.RandomCrop(size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()

    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_content = (
        (content_feat - content_mean.expand(size))
        / content_std.expand(size)
    )

    return normalized_content * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5):
    assert feat.dim() == 4

    batch_size, channels = feat.size()[:2]

    feat = feat.view(batch_size, channels, -1)

    feat_mean = feat.mean(dim=2).view(batch_size, channels, 1, 1)

    feat_std = (
        feat.var(dim=2, unbiased=False) + eps
    ).sqrt().view(batch_size, channels, 1, 1)

    return feat_mean, feat_std