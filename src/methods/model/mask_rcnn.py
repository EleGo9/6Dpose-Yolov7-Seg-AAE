from torch import nn, load, cat
from torch.nn import *

from torchvision.models.detection.mask_rcnn import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNNModel:
    STANDARD_MASK_RCNN = "STANDARD_MASK_R_CNN"
    U_MASK_RCNN = "U_MASK_R_CNN"
    EXTENDED_U_MASK_RCNN = "EXTENDED_U_MASK_R_CNN"

    def __init__(self, num_classes: int, mask_predictor: str = STANDARD_MASK_RCNN):
        self.num_classes = num_classes
        self.mask_predictor = mask_predictor

        self.model = None

        self.initialize()

        print(self.model)

    def initialize(self):
        self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels

        if self.mask_predictor == self.STANDARD_MASK_RCNN:
            print("Standard Mask R-CNN predictor")
            hidden_layer = 256
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask,
                hidden_layer,
                self.num_classes
            )
        elif self.mask_predictor == self.U_MASK_RCNN:
            print("U-Mask R-CNN predictor")
            self.model.roi_heads.mask_predictor = UMaskRCNNPredictor(
                in_features_mask,
                self.num_classes
            )
        elif self.mask_predictor == self.EXTENDED_U_MASK_RCNN:
            print("Extended U-Mask R-CNN predictor")
            self.model.roi_heads.mask_predictor = ExtendedUMaskRCNNPredictor(
                in_features_mask,
                self.num_classes
            )

    def load_weights(self, weights_path):
        self.model.load_state_dict(load(weights_path))

    def to_device(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()

    def segment(self, x):
        return self.model([x])


class UMaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv_1 = None
        self.transp_conv_1 = None
        self.transp_conv_2 = None
        self.conv_2 = None

        self.initialize()

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def initialize(self):
        in_c, out_c = self.in_channels, self.in_channels
        self.conv_1 = Convolution(in_c, out_c)  # 256, 7, 7
        in_c, out_c = out_c, out_c//2
        self.transp_conv_1 = TransposeConvolution(in_c, out_c)  # 128, 14, 14
        in_c, out_c = out_c, out_c//2
        self.transp_conv_2 = TransposeConvolution(in_c, out_c)  # 64, 28, 28
        in_c, out_c = out_c, self.num_classes
        self.conv_2 = Convolution(in_c, out_c, one_by_one=True)  # classes, 28, 28
        # self.conv_2 = Conv2d(in_c, out_c, kernel_size=1, stride=1, padding="same")

    def forward(self, x):
        # print("input", x.size())
        y = self.conv_1(x)
        # print("conv_1", y.size())
        y = self.transp_conv_1(y)
        # print("transp_conv_1", y.size())
        y = self.transp_conv_2(y)
        # print("transp_conv_2", y.size())
        y = self.conv_2(y)
        # print("conv_2", y.size())
        return y


class ExtendedUMaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv_1 = None
        self.conv_2 = None
        self.transp_conv_1 = None
        self.transp_conv_2 = None
        self.conv_3 = None
        self.conv_4 = None

        self.initialize()

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def initialize(self):
        in_c, out_c = self.in_channels, self.in_channels//2
        self.conv_1 = Convolution(in_c, out_c, max_pool=False)  # 128, 14, 14
        in_c, out_c = self.in_channels, self.in_channels
        self.conv_2 = Convolution(in_c, out_c)  # 256, 7, 7
        in_c, out_c = out_c, out_c//2
        self.transp_conv_1 = TransposeConvolution(in_c, out_c)  # 128, 14, 14

        # 256, 14, 14
        in_c, out_c = out_c*2, out_c
        self.transp_conv_2 = TransposeConvolution(in_c, out_c)  # 128, 28, 28
        in_c, out_c = out_c, out_c//2
        self.conv_3 = Convolution(in_c, out_c, max_pool=False)  # 64, 28, 28
        in_c, out_c = out_c, self.num_classes
        self.conv_4 = Convolution(in_c, out_c, one_by_one=True)  # classes, 28, 28

    def forward(self, x):
        # print("input", x.size())

        y_1 = self.conv_1(x)
        # print("conv_1", y_1.size())

        y_2 = self.conv_2(x)
        # print("conv_2", y_2.size())
        y_2 = self.transp_conv_1(y_2)
        # print("transp_conv_1", y_2.size())

        y = self.transp_conv_2(y_1, y_2)
        # print("transp_conv_2", y.size())
        y = self.conv_3(y)
        # print("conv_3", y.size())
        y = self.conv_4(y)
        # print("conv_4", y.size())
        return y


class Convolution(Module):
    def __init__(self, in_channels: int, out_channels: int, max_pool: bool = True, one_by_one: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_pool = max_pool
        self.one_by_one = one_by_one

        self.conv = None

        self.initialize()

    def initialize(self):
        if self.one_by_one:
            self.conv = Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding="same")
            return

        if self.max_pool:
            self.conv = Sequential(
                Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding="same"),
                ReLU(inplace=True),
                MaxPool2d(2, 2)
            )
        else:
            self.conv = Sequential(
                Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding="same"),
                ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class TransposeConvolution(Module):
    def __init__(self, in_channels: int, out_channels: int, skip_connection: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_connection = skip_connection

        self.conv_transpose = None

        self.initialize()

    def initialize(self):
        self.conv_transpose = Sequential(
            nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2, padding=0),
            ReLU(inplace=True)
        )

    def forward(self, x, x_s=None):
        if x_s is not None:
            x = cat([x, x_s], dim=1)
        return self.conv_transpose(x)
