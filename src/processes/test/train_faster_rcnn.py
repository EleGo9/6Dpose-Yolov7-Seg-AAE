import os
import sys

import torch

sys.path.append(os.getcwd())
from src.methods.model.faster_rcnn import FasterRCNNModel
from src.services.methods.train.train_faster_rcnn import TrainFasterRCNN
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FasterRCNNModel(num_classes=4)
model = model.return_model()

train = TrainFasterRCNN("src/config/train_faster_rcnn.yml", device)
train.define_transform()
train.define_dataset()
train.define_dataloader()
train.define_model(model)
train.define_settings()
train.fit()

print("Train finished")
