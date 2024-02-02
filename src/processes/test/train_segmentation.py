import os
import sys

import torch

sys.path.append(os.getcwd())
from src.methods.model.sg_det import SGDet
from src.methods.model.custom_model import CustomModel
from src.methods.model.unet import UNet
from src.services.methods.train.train_segmentation import TrainSegmentation
from torchvision.models.segmentation import deeplabv3_resnet50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = deeplabv3_resnet50(pretrained=True)
#model.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
model = SGDet(num_channels=3, num_classes=4, weights="src/weights/segmentation/sg_l_final_new_001.torch")
# model = CustomModel(num_channels=3, num_classes=3)
# model = UNet(num_channels=3, num_classes=3)

train = TrainSegmentation("src/config/train_segmentation.yml", device)
train.define_transform()
train.define_dataset()
train.define_dataloader()
train.define_model(model)
train.define_settings()
train.fit()

print("Train finished")
