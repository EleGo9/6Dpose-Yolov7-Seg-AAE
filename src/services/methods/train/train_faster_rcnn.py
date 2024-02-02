import yaml

import numpy as np
import cv2
import torch
from torch.nn import *
from torchvision.transforms import *
from torch.optim import *
from torch.cuda import empty_cache

from src.methods.dataset.faster_rcnn_dataset import DatasetFasterRCNN
from src.methods.dataloader.custom_dataloader import CustomDataloader

import albumentations as alb
import albumentations.pytorch as albtorch


class TrainFasterRCNN:
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
        self.optimizer = None
        self.criterion = None

        self.initialize()

    def initialize(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)
        print(self.config)

    def define_transform(self):
        self.transform_image = Compose([
            ToPILImage(),
            Resize(self.config["transform"]["image_size"]),
            ToTensor(),
            #RandomErasing(p=0.5, scale=(0.1, 0.1), ratio=(0.1, 0.1)),
            #ColorJitter((0.5, 1), (0.5, 1), (0.5, 1), (-0.1, 0.1)),
            #GaussianBlur(kernel_size=3, sigma=0.3)
        ])
        self.transform_mask = Compose([
            ToPILImage(),
            Grayscale(),
            Resize(self.config["transform"]["image_size"], InterpolationMode.NEAREST),
            ToTensor()
        ])

    def define_dataset(self):
        self.dataset = DatasetFasterRCNN(
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

    def define_model(self, model):
        self.model = model.to(self.device)
        self.model.train()

    def define_settings(self):
        self.optimizer = SGD(
            params=self.model.parameters(),
            lr=0.001,
            momentum=0.9,
            weight_decay=0.0005
        )
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=1e-5
        )
        self.criterion = CrossEntropyLoss()

    def fit(self):
        num_epochs = self.config["hyper_parameters"]["num_epochs"]
        for epoch in range(num_epochs):
            for iteration, (index, images, masks, targets) in enumerate(self.dataloader):
                images = list(image.to(self.device).requires_grad_(False) for image in images)
                #masks = list(mask.squeeze(0).to(self.device).requires_grad_(False) for mask in masks)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()

                '''from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
                bb = targets[0]["boxes"]
                m = targets[0]["masks"].type(torch.bool)
                bounding_box = draw_bounding_boxes((images[0] * 255).type(torch.uint8), bb, colors="red")
                im = draw_segmentation_masks(bounding_box, m, alpha=0.3, colors=["red", "green"])
                im = im.cpu().numpy().transpose(1, 2, 0)
                cv2.putText(im, str(targets[0]["labels"][0].item()), (int(targets[0]["boxes"][0][0]), int(targets[0]["boxes"][0][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, 1, 2)
                cv2.imshow("image", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                cv2.waitKey(10)'''

                '''from torchvision.utils import draw_bounding_boxes
                self.model.eval()
                prediction = self.model(images)
                image = images[0]
                bb = prediction[0]["boxes"]
                print(bb)
                bounding_box = draw_bounding_boxes((image*255).type(torch.uint8), bb, colors="red")
                image = (image*255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
                cv2.imshow("image", cv2.cvtColor(bounding_box.cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB))
                cv2.waitKey(0)'''

                if epoch > num_epochs - 2:
                    pass # self.show(images, masks, targets, prediction)
                self.print_status(epoch, iteration, losses)

                for image in images:
                    image.detach()
                losses.detach()
                del images, losses
                empty_cache()

            self.save_model()
        self.save_model()

    def show(self, image, label, prediction):
        show_interval = self.config["utils"]["show"]
        if show_interval is not None:
            image_show = image[0].cpu().detach().numpy().transpose(1, 2, 0)
            label_show = label[0].cpu().detach().numpy()
            label_show = np.dstack((label_show, np.zeros_like(label_show), np.zeros_like(label_show)))
            prediction_show = np.where(prediction == 1, 255, 0)
            prediction_show = prediction_show.astype(np.uint8)
            prediction_show = np.dstack((prediction_show, np.zeros_like(prediction_show), np.zeros_like(prediction_show)))
            full_show = np.concatenate((image_show, label_show, prediction_show), axis=1)
            cv2.imshow("image - mask - prediction", cv2.cvtColor(full_show, cv2.COLOR_BGR2RGB))
            cv2.waitKey(show_interval)

    def print_status(self, epoch, iteration, loss):
        print("Epoch {:03d}/{:03d} ({:04d}-th iteration), loss: {:.7f}".format(
            epoch + 1,
            self.config["hyper_parameters"]["num_epochs"],
            iteration + 1,
            loss.item()
        ))

    def save_model(self):
        name = self.config["general"]["save_model_path"]
        print("Saving model {}.torch".format(name))
        torch.save(self.model.state_dict(), "{}.torch".format(name))
