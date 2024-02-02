import os
import sys

import torch

sys.path.append(os.getcwd())
from src.methods.model.custom_model import CustomModel
from src.methods.model.unet import UNet
from src.services.methods.train.train import Train


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CustomModel(num_channels=3, num_classes=2)
# model = UNet(num_channels=3, num_classes=2, bilinear=False)

train = Train("src/config/train.yml", device)
train.define_transform()
train.define_dataset()
train.define_dataloader()
train.define_model(model)

train.define_settings()
train.fit()
print("Train finished")
