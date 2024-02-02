import yaml

import numpy as np
import cv2
import torch
from torch.nn import *
from torchvision.transforms import *
from torch.optim import *
from torch.cuda import empty_cache
from torch.autograd import Variable
from torcheval.metrics import *
from torchvision.ops import box_iou, nms
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

from src.methods.dataset.segmentation_dataset import SegmentationDataset
from src.methods.dataloader.custom_dataloader import CustomDataloader

import albumentations as alb
import albumentations.pytorch as albtorch


class MetricsDetection:
    def __init__(
            self,
            config_path: str,
            device: torch.device = torch.device("cuda:0")
    ):
        self.config_path = config_path
        self.device = device

        self.config = None
        self.transform_image = None
        self.transform_mask = None
        self.dataset = None
        self.dataloader = None
        self.model = None

        self.metrics = None

        self.initialize()

    def initialize(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)
        print(self.config)

    def define_transform(self):
        self.transform_image = Compose([
            ToPILImage(),
            Resize(self.config["transform"]["image_size"]),
            ToTensor()
        ])
        self.transform_mask = Compose([
            ToPILImage(),
            Grayscale(),
            Resize(self.config["transform"]["image_size"], InterpolationMode.NEAREST),
            ToTensor()
        ])

    def define_dataset(self):
        self.dataset = SegmentationDataset(
            self.config["general"]["root_path"],
            transform_image=self.transform_image,
            transform_mask=self.transform_mask
        )

    def define_dataloader(self):
        self.dataloader = CustomDataloader(
            dataset=self.dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=self.config["hyper_parameters"]["shuffle"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            collate_fn=lambda batch: list(zip(*batch))
        )

    def define_model(self, model: Module):
        self.model = model.to(self.device)
        model.load_state_dict(torch.load(self.config["general"]["model_path"]))
        self.model.eval()

    @torch.no_grad()
    def compute(self):
        for iteration, (index, images, masks, targets) in enumerate(self.dataloader):
            images = list(image.to(self.device).requires_grad_(False) for image in images)
            # masks = list(mask.squeeze(0).to(self.device).requires_grad_(False) for mask in masks)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            images = torch.stack(images)
            predictions = self.model(images)

            for prediction, target, image in zip(predictions, targets, images):
                boxes = prediction["boxes"]
                scores = prediction["scores"]
                bb_indexing = scores > 0.8
                boxes = boxes[bb_indexing]
                scores = scores[bb_indexing]
                non_max_suppression = nms(boxes, scores, 0.1)
                boxes = boxes[non_max_suppression]
                scores = scores[non_max_suppression]
                labels = [str(i.item()) for i in prediction["labels"][non_max_suppression]]

                iou_scores = self.compute_iou(boxes, target["boxes"])
                self.print_status(iteration, iou_scores, scores, labels)
                self.show(image, boxes, target["boxes"])

            for image in images:
                image.detach()
            del images, predictions
            empty_cache()

    def show(self, image, prediction, boxes):
        image = draw_bounding_boxes((image * 255).type(torch.uint8), boxes, colors="green")
        image = draw_bounding_boxes(image, prediction, colors="red")
        cv2.imshow("prediction - boxes", cv2.cvtColor(image.cpu().detach().numpy().transpose((1, 2, 0)), cv2.COLOR_BGR2RGB))
        cv2.waitKey()

    def print_status(self, iteration, iou_scores, scores, labels):
        print("{:04d}-th iteration, iou score: ".format(
            iteration + 1
        ), end="")
        for iou_score in iou_scores:
            iou_score = torch.max(iou_score)
            print("{:.7f}".format(iou_score.item()), end=" ")
        if len(iou_scores) < 3:
            for i in range(3 - len(iou_scores)):
                print("{:.7f}".format(0.0), end=" ")
        print(", labels [scores]: ", end="")
        for label, score in zip(labels, scores):
            print("{} [{:.7f}]".format(label, score), end=" ")
        print(flush=True)

    def compute_iou(self, prediction, boxes):
        iou_score = box_iou(prediction, boxes)

        return iou_score
