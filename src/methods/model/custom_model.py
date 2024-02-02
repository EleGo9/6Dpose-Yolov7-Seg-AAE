import torch
from torch import cat
from torch.nn import *
import torch.nn.functional as F


class CustomModel(Module):
    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.inc = None
        self.down_1 = None
        self.down_2 = None
        self.down_3 = None
        self.down_4 = None
        self.up_1 = None
        self.up_2 = None
        self.up_3 = None
        self.up_4 = None
        self.out = None

        self.initialize()

    def initialize(self):
        self.inc = DoubleConv(self.num_channels, 16, 16)

        self.down_1 = Down(16, 32, 64)
        self.down_2 = Down(64, 128, 256)
        self.down_3 = Down(256, 512, 1024)

        self.up_1 = Up(1024, 512, 256, True)
        self.up_2 = Up(256, 128, 64)
        self.up_3 = Up(64, 32, 16)

        self.out = OutConv(16, self.num_classes)

    def forward(self, x):
        x_1 = self.inc(x)
        #print(x_1.size())

        x_2 = self.down_1(x_1)
        #print(x_2.size())
        x_3 = self.down_2(x_2)
        #print(x_3.size())
        x_4 = self.down_3(x_3)
        #print(x_4.size())

        x = self.up_1(x_4, x_3)
        #print(x.size())
        x = self.up_2(x, x_2)
        #print(x.size())
        x = self.up_3(x, x_1)
        #print(x.size())

        out = self.out(x)
        #print(out.size())
        return out

    def total_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DoubleConv(Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.conv = None

        self.initialize()

    def initialize(self):
        self.conv = Sequential(
            Conv2d(self.in_channels, self.mid_channels, kernel_size=3, padding="same"),
            BatchNorm2d(self.mid_channels),
            ReLU(inplace=True),

            Conv2d(self.mid_channels, self.out_channels, kernel_size=3, padding="same"),
            BatchNorm2d(self.out_channels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.max_pool = None

        self.initialize()

    def initialize(self):
        self.max_pool = Sequential(
            DoubleConv(self.in_channels, self.mid_channels, self.out_channels),
            MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.max_pool(x)


class Up(Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, bridge: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.bridge = bridge

        self.up_sample = None
        self.conv = None

        self.initialize()

    def initialize(self):
        if not self.bridge:
            self.up_sample = Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(self.in_channels, self.mid_channels, self.out_channels)
        else:
            self.up_sample = ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(self.mid_channels, self.mid_channels, self.out_channels)

    def forward(self, x_1, x_2):
        #print(x_1.size(), x_2.size())
        x_1 = self.up_sample(x_1)
        #print(x_1.size(), x_2.size())
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


