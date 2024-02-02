import yaml

import numpy as np
import cv2
import torch
from torch.nn import *
from torchvision.transforms import *
from torch.optim import *
from torch.cuda import empty_cache
from torch.autograd import Variable

from src.methods.dataset.segmentation_dataset import SegmentationDataset
from src.methods.dataloader.custom_dataloader import CustomDataloader

import albumentations as alb
import albumentations.pytorch as albtorch


class TrainSegmentation:
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
            #RandomErasing(p=0.5, scale=(0.05, 0.05), ratio=(0.2, 0.7)),
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
        self.model.train()

    def define_settings(self):
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.config["hyper_parameters"]["learning_rate"],
            weight_decay=self.config["hyper_parameters"]["weight_decay"],
            betas=self.config["hyper_parameters"]["betas"]
        )
        self.criterion = CrossEntropyLoss()

    def fit(self):
        num_epochs = self.config["hyper_parameters"]["num_epochs"]
        for epoch in range(num_epochs):
            for iteration, (index, images, masks, targets) in enumerate(self.dataloader):
                #images = Variable(torch.stack(images), requires_grad=False).to(self.device)
                #masks = Variable(torch.stack(masks), requires_grad=False).to(self.device).squeeze(1)

                images = list(image.to(self.device).requires_grad_(False) for image in images)
                masks = list(mask.squeeze(0).to(self.device).requires_grad_(False) for mask in masks)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.model.zero_grad()
                images = torch.stack(images)
                predictions = self.model(images)
                masks = torch.stack(masks)
                loss = self.criterion(predictions, masks.long())
                loss.backward()
                self.optimizer.step()

                if epoch > num_epochs - 2:
                    # self.show(images, masks, targets, prediction)
                    prediction = torch.argmax(predictions, 1).detach().cpu()
                    prediction[prediction == 1] = 20
                    prediction[prediction == 2] = 120
                    prediction[prediction == 3] = 255
                    prediction = prediction.type(torch.uint8)

                    masks = masks.detach().cpu()
                    masks[masks == 1] = 20
                    masks[masks == 2] = 120
                    masks[masks == 3] = 255
                    masks = masks.type(torch.uint8)

                    im_1 = np.concatenate((prediction[0], masks[0]), axis=1)
                    im_2 = np.concatenate((prediction[1], masks[1]), axis=1)
                    full_show = np.concatenate((im_1, im_2), axis=0)
                    cv2.imshow("full_show", full_show)
                    cv2.waitKey(1)
                self.print_status(epoch, iteration, loss)

                images.detach()
                masks.detach()
                loss.detach()
                predictions.detach()
                del images, masks, loss, predictions
                empty_cache()

            self.save_model()
        self.save_model()

    def show(self, images, masks, targets, prediction):
        show_interval = self.config["utils"]["show"]
        if show_interval is not None:
            # for image, mask, target in zip(images, masks, targets):
            image_show = images[0].detach().cpu().numpy().transpose(1, 2, 0)
            masks[masks != 0] = 255
            label_show = masks[0].detach().cpu().numpy()
            prediction_show = prediction[0].detach().cpu().numpy()
            label_show = np.dstack((label_show, np.zeros_like(label_show), np.zeros_like(label_show)))
            prediction_show = np.where(prediction_show != 0, 255, 0)
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
            loss.data.cpu().numpy()
        ))

    def save_model(self):
        name = self.config["general"]["save_model_path"]
        print("Saving model {}.torch".format(name))
        torch.save(self.model.state_dict(), "{}.torch".format(name))
