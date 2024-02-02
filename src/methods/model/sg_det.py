import torch
from torch import cat
from torch.nn import *
from torchvision.transforms import Resize
from torch.nn.functional import pad
from torch import onnx


class SGDet(Module):
    _NAME = "SGDet"

    def __init__(self, num_channels: int, num_classes: int, weights: str = None):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.weights = weights

        self.resize = None

        self.inc = None
        self.down_1 = None
        self.down_2 = None
        self.down_3 = None
        self.down_4 = None
        self.down_5 = None
        self.down_6 = None
        self.up_1 = None
        self.up_2 = None
        self.up_3 = None
        self.up_4 = None
        self.up_5 = None
        self.up_6 = None
        self.out = None

        self.initialize()

    def initialize(self):
        self.resize = Resize((128, 128))

        self.inc = BaseConv(self.num_channels, 16)

        self.down_1 = Down(16, 32)
        self.down_2 = Down(32, 64)
        self.down_3 = Down(64, 128)
        self.down_4 = Down(128, 256)
        self.down_5 = Down(256, 512)
        self.down_6 = Down(512, 1024)

        self.up_1 = Up(1024, 512)
        self.up_2 = Up(512, 256)
        self.up_3 = Up(256, 128)
        self.up_4 = Up(128, 64)
        self.up_5 = Up(64, 32)
        self.up_6 = Up(32, 16)

        self.out = OutConv(16, self.num_classes)

        if self.weights is not None:
            self.load_state_dict(torch.load(self.weights))

        self.print_net()

    def print_net(self):
        print("{}:".format(self._NAME))
        print("\t - input channels: {}".format(self.num_channels))
        print("\t - classes: {}".format(self.num_classes))
        if self.weights is not None:
            print("\t - pre-trained weights: {}".format(self.weights))
        print("\t - total parameters: {}".format(self.total_parameters()))

    def total_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx(self, image: torch.Tensor):
        image = image.unsqueeze(0).cuda()
        print(image.size())
        onnx.export(
            model=self,
            args=image,
            f="src/arch/sg_l_arch.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=16
        )

    def forward(self, x):
        #print(x.size())
        # self.resize(x)
        # print(x.size())
        x_1 = self.inc(x)
        #print(x_1.size())

        x_2 = self.down_1(x_1)
        #print(x_2.size())
        x_3 = self.down_2(x_2)
        #print(x_3.size())
        x_4 = self.down_3(x_3)
        #print(x_4.size())
        x_5 = self.down_4(x_4)
        #print(x_5.size())
        x_6 = self.down_5(x_5)
        #print(x_6.size())
        x_7 = self.down_6(x_6)
        #print("encoder:", x_7.size())

        x = self.up_1(x_7, x_6)
        #print(x.size())
        x = self.up_2(x, x_5)
        #print(x.size())
        x = self.up_3(x, x_4)
        #print(x.size())
        x = self.up_4(x, x_3)
        #print(x.size())
        x = self.up_5(x, x_2)
        #print(x.size())
        x = self.up_6(x, x_1)
        #print(x.size())

        out = self.out(x)
        #print("decoder:", out.size())
        return out


class BaseConv(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = None

        self.initialize()

    def initialize(self):
        self.conv = Sequential(
            Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding="same"),
            BatchNorm2d(self.out_channels),
            ReLU(inplace=True),

            Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding="same"),
            BatchNorm2d(self.out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class Down(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.max_pool = None

        self.initialize()

    def initialize(self):
        self.max_pool = Sequential(
            BaseConv(self.in_channels, self.out_channels),
            MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.max_pool(x)


class Up(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_transpose = None
        self.conv = None

        self.initialize()

    def initialize(self):
        self.conv_transpose = ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2)
        self.conv = BaseConv(self.in_channels, self.out_channels)

    def forward(self, x_1, x_2):
        x_1 = self.conv_transpose(x_1)
        diff_x = x_2.size()[3] - x_1.size()[3]
        diff_y = x_2.size()[2] - x_1.size()[2]
        x_1 = pad(
            x_1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            mode="replicate"
        )
        x = cat([x_1, x_2], dim=1)
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
