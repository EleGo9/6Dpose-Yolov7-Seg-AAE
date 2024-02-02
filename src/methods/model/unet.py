import torch
from torch import cat
from torch.nn import *
import torch.nn.functional as F


class UNet(Module):
    def __init__(self, num_channels: int, num_classes: int, bilinear: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = None
        self.down_1 = None
        self.down_2 = None
        self.down_3 = None
        self.factor = 2 if bilinear else 1
        self.down_4 = None
        self.up_1 = None
        self.up_2 = None
        self.up_3 = None
        self.up_4 = None
        self.out = None

        self.initialize()

    def initialize(self):
        self.inc = DoubleConv(self.num_channels, 64)
        self.down_1 = Down(64, 128)
        self.down_2 = Down(128, 256)
        self.down_3 = Down(256, 512)
        self.down_4 = Down(512, 1024 // self.factor)
        self.up_1 = Up(1024, 512 // self.factor, self.bilinear)
        self.up_2 = Up(512, 256 // self.factor, self.bilinear)
        self.up_3 = Up(256, 128 // self.factor, self.bilinear)
        self.up_4 = Up(128, 64, self.bilinear)
        self.out = OutConv(64, self.num_classes)

    def forward(self, x):
        x_1 = self.inc(x)
        x_2 = self.down_1(x_1)
        x_3 = self.down_2(x_2)
        x_4 = self.down_3(x_3)
        x_5 = self.down_4(x_4)
        x = self.up_1(x_5, x_4)
        x = self.up_2(x, x_3)
        x = self.up_3(x, x_2)
        x = self.up_4(x, x_1)
        out = self.out(x)
        return out


class DoubleConv(Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        self.conv = None

        self.initialize()

    def initialize(self):
        if not self.mid_channels:
            self.mid_channels = self.out_channels
        self.conv = Sequential(
            Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding="same"),
            BatchNorm2d(self.mid_channels),
            ReLU(inplace=True),
            Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding="same"),
            BatchNorm2d(self.mid_channels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.maxpool = None

        self.initialize()

    def initialize(self):
        self.maxpool = Sequential(
            MaxPool2d(2),
            DoubleConv(self.in_channels, self.out_channels)
        )

    def forward(self, x):
        return self.maxpool(x)


class Up(Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.up = None
        self.conv = None

        self.initialize()

    def initialize(self):
        if self.bilinear:
            self.up = Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(self.in_channels, self.out_channels, self.in_channels // 2)
        else:
            self.up = ConvTranspose2d(self.in_channels, self.in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(self.in_channels, self.out_channels)

    def forward(self, x_1, x_2):
        x_1 = self.up(x_1)
        diff_x = x_2.size()[2] - x_1.size()[2]
        diff_y = x_2.size()[3] - x_1.size()[3]
        x_1 = F.pad(
            x_1,
            [
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2
            ]
        )
        x = cat([x_2, x_1], dim=1)
        return self.conv(x)


class OutConv(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = None

        self.initialize()

    def initialize(self):
        self.conv = Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding="same")

    def forward(self, x):
        return self.conv(x)


