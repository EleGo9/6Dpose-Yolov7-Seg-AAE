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

from src.methods.dataset.segmentation_dataset import SegmentationDataset
from src.methods.dataloader.custom_dataloader import CustomDataloader

import albumentations as alb
import albumentations.pytorch as albtorch


class MetricsSegmentation:
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
        from statistics import mean, variance, pstdev
        confidence_scores_obj_1 = []
        confidence_scores_obj_2 = []
        confidence_scores_obj_3 = []
        for iteration, (index, images, masks, targets) in enumerate(self.dataloader):
            images = list(image.to(self.device).requires_grad_(False) for image in images)
            masks = list(mask.squeeze(0).to(self.device).requires_grad_(False) for mask in masks)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            images = torch.stack(images)
            predictions = self.model(images)
            predictions = torch.argmax(predictions, 1)
            masks = torch.stack(masks)

            for prediction, mask, target in zip(predictions, masks, targets):
                label = target["labels"][0]
                prediction[prediction != label] = 0

                iou_score = self.compute_iou(prediction, mask)
                self.print_status(iteration, iou_score)

                # self.show(prediction, mask)
                print(iou_score)

                if label == 1:
                    confidence_scores_obj_1.append(iou_score.item())
                elif label == 2:
                    confidence_scores_obj_2.append(iou_score.item())
                elif label == 3:
                    confidence_scores_obj_3.append(iou_score.item())
            images.detach()
            masks.detach()
            predictions.detach()
            del images, masks, predictions
            empty_cache()

        print(len(confidence_scores_obj_1) + len(confidence_scores_obj_2) + len(confidence_scores_obj_1))

        print(mean(confidence_scores_obj_1), sum(confidence_scores_obj_1) / len(confidence_scores_obj_1))
        print(variance(confidence_scores_obj_1), np.var(confidence_scores_obj_1))

        print(mean(confidence_scores_obj_2), sum(confidence_scores_obj_2) / len(confidence_scores_obj_2))
        print(variance(confidence_scores_obj_2))

        print(mean(confidence_scores_obj_3), sum(confidence_scores_obj_3) / len(confidence_scores_obj_3))
        print(variance(confidence_scores_obj_3))

    def show(self, prediction, mask):
        full_show = np.concatenate((prediction.cpu().detach().numpy(), mask.cpu().detach().numpy()), axis=1)
        cv2.imshow("prediction - mask", full_show)
        cv2.waitKey()

    def print_status(self, iteration, iou_score):
        print("{:04d}-th iteration, iou score: {:.7f}".format(
            iteration + 1,
            iou_score
        ))

    def compute_iou(self, prediction, masks):
        intersection = torch.logical_and(prediction, masks)
        union = torch.logical_or(prediction, masks)
        iou_score = torch.sum(intersection) / torch.sum(union)

        return iou_score
