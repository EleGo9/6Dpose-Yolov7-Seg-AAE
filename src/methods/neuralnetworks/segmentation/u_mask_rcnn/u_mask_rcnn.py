import time
import yaml
import numpy as np
import cv2
import torch
from torch.nn import *
from torchvision.transforms import *
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.transforms.functional import *

from src.methods.neuralnetworks.segmentation.interface_segmentation import ISegmentation
from src.methods.model.mask_rcnn import MaskRCNNModel


class UMaskRCNN(ISegmentation):
    def __init__(self, config_path, debug_vis=False):
        self.config_path = config_path
        self.debug_vis = debug_vis

        self.config = None
        self.device = None
        self.model = None
        self.transform_image = None
        self.threshold = 0.8

        self.initialize()

    def initialize(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)
        self.load_model()
        self.define_transform()
        self.threshold = self.config["model"]["threshold"]

    def load_model(self):
        version = MaskRCNNModel.STANDARD_MASK_RCNN
        if self.config["model"]["version"] == "u_mask_rcnn":
            version = MaskRCNNModel.U_MASK_RCNN
        elif self.config["model"]["version"] == "extended_u_mask_rcnn":
            version = MaskRCNNModel.EXTENDED_U_MASK_RCNN
        self.model = MaskRCNNModel(num_classes=self.config["model"]["num_classes"], mask_predictor=version)
        print(self.config["model"]["weights"])
        self.model.load_weights(self.config["model"]["weights"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to_device(self.device)
        self.model.eval()

    def define_transform(self):
        self.transform_image = Compose([
            ToPILImage(),
            ToTensor(),
        ])

    def segment(self, image):
        image = self.transform_image(image)
        image = image.to(self.device)
        with torch.no_grad():
            prediction = self.model.segment(image)

        labels = prediction[0]["labels"]
        scores = prediction[0]["scores"]
        boxes = prediction[0]["boxes"]
        masks = prediction[0]["masks"]

        threshold_indexing = scores > self.threshold
        labels = labels[threshold_indexing]
        scores = scores[threshold_indexing]
        boxes = boxes[threshold_indexing]
        masks = masks[threshold_indexing]

        masks = masks.squeeze(1)
        masks = masks > 0.5

        return labels.cpu().numpy(), scores.cpu().numpy(), boxes.cpu().numpy(), masks.cpu().numpy()

    def draw(self, image, labels, scores, boxes, masks):
        image = torch.from_numpy(image.transpose(2, 0, 1)).type(torch.uint8)
        scores = torch.from_numpy(scores)
        boxes = torch.from_numpy(boxes)
        masks = torch.from_numpy(masks)
        labels = ["{}: {:.3f}".format(l.item(), s.item()) for l, s in zip(labels, scores)]
        colors = ["red", "green", "blue", "orange", "white", "purple", "yellow", "aqua", "black", "brown", "red", "green", "blue", "orange", "white", "purple", "yellow", "aqua", "black", "brown"]
        bounding_box = draw_bounding_boxes(image, boxes, colors=colors[:masks.size()[0]]) #, labels=labels)
        masked_image = draw_segmentation_masks(bounding_box, masks, alpha=0.3, colors=colors[:masks.size()[0]])
        masked_image = masked_image.cpu().numpy().transpose(1, 2, 0)
        for l, b in zip(labels, boxes):
            cv2.putText(masked_image, l,
                        (int(b[0]), int(b[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (30, 200, 255), 2)
        # cv2.imshow("image", cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        # cv2.waitKey()
        return masked_image

    def mask(self, image, masks):
        masks = masks.transpose(1, 2, 0)
        masks = masks.max(axis=2)
        masks = np.expand_dims(masks, axis=2)
        masks = masks.astype(np.uint8)

        masked_image = cv2.bitwise_and(image, image, mask=masks)
        return masked_image
