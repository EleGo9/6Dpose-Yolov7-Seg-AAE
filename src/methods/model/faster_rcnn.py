from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import *


class FasterRCNNModel:
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        self.model = None

        self.initialize()

    def initialize(self):
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

    def return_model(self):
        return self.model
