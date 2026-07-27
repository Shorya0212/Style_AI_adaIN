import torch
import torch.nn as nn


class VGGEncoder(nn.Module):
    def __init__(self, vgg_path):
        super(VGGEncoder, self).__init__()

        self.vgg = nn.Sequential(
            # ---------------- Block 1 ----------------
            nn.Conv2d(3, 3, (1, 1)),                            # RGB -> BGR
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),                           # conv1_1
            nn.ReLU(),                                          # relu1_1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),                          # conv1_2
            nn.ReLU(),                                          # relu1_2

            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # pool1

            # ---------------- Block 2 ----------------
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),                         # conv2_1
            nn.ReLU(),                                          # relu2_1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),                        # conv2_2
            nn.ReLU(),                                          # relu2_2

            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # pool2

            # ---------------- Block 3 ----------------
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),                        # conv3_1
            nn.ReLU(),                                          # relu3_1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),                        # conv3_2
            nn.ReLU(),                                          # relu3_2

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),                        # conv3_3
            nn.ReLU(),                                          # relu3_3

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),                        # conv3_4
            nn.ReLU(),                                          # relu3_4

            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # pool3

            # ---------------- Block 4 ----------------
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),                        # conv4_1
            nn.ReLU(),                                          # relu4_1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),                        # conv4_2
            nn.ReLU(),                                          # relu4_2

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),                        # conv4_3
            nn.ReLU(),                                          # relu4_3

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),                        # conv4_4
            nn.ReLU(),                                          # relu4_4

            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # pool4

            # ---------------- Block 5 ----------------
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),                        # conv5_1
            nn.ReLU(),                                          # relu5_1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),                        # conv5_2
            nn.ReLU(),                                          # relu5_2

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),                        # conv5_3
            nn.ReLU(),                                          # relu5_3

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),                        # conv5_4
            nn.ReLU()                                           # relu5_4
        )

        # Load pretrained VGG weights
        self.vgg.load_state_dict(torch.load(vgg_path, map_location="cpu"))

        # Keep encoder till relu4_1
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        enc_layers = list(self.vgg.children())

        self.enc_1 = nn.Sequential(*enc_layers[:4])      # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])    # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])   # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])   # relu3_1 -> relu4_1

        # Freeze encoder
        for name in ["enc_1", "enc_2", "enc_3", "enc_4"]:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def forward(self, input, is_test=False):
        h1 = self.enc_1(input)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)

        if is_test:
            return h4

        return [h1, h2, h3, h4]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, input):
        return self.decoder(input)