import yaml

import numpy as np
import cv2
import torch
from torch.nn import *
from torchvision.transforms import *
from torch.optim import *

from src.methods.dataset.custom_dataset import CustomDataset
from src.methods.dataloader.custom_dataloader import CustomDataloader

import albumentations as alb
import albumentations.pytorch as albtorch


class Train:
    def __init__(
            self,
            config_path: str,
            device: torch.device = torch.device("cuda:0")
    ):
        self.config_path = config_path
        self.device = device

        self.config = None
        self.transform_image = None
        self.transform_label = None
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
            RandomErasing(p=0.5, scale=(0.1, 0.1), ratio=(0.1, 0.1)),
            ColorJitter((0.5, 1), (0.5, 1), (0.5, 1), (-0.1, 0.1)),
            GaussianBlur(kernel_size=3, sigma=0.3)
        ])
        self.transform_label = Compose([
            ToPILImage(),
            Grayscale(),
            Resize(self.config["transform"]["image_size"], InterpolationMode.NEAREST),
            ToTensor()
        ])

    def define_augmented_transform(self):
        self.transform_image = alb.Compose([
            alb.Resize(
                self.config["transform"]["image_size"][0],
                self.config["transform"]["image_size"][1]
            )
        ])
        self.transform_label = Compose([
            ToPILImage(),
            Grayscale(),
            Resize(self.config["transform"]["image_size"], InterpolationMode.NEAREST),
            ToTensor()
        ])

    def define_dataset(self):
        self.dataset = CustomDataset(
            self.config["general"]["images_labels_path"],
            self.config["general"]["root_path"],
            images_dir="rgb/",
            labels_dir="mask/",
            transform_image=self.transform_image,
            transform_label=self.transform_label
        )

    def define_dataloader(self):
        self.dataloader = CustomDataloader(
            dataset=self.dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=self.config["hyper_parameters"]["shuffle"],
            num_workers=self.config["hyper_parameters"]["num_workers"]
            #collate_fn=lambda x: list(zip(*x))
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
        for e in range(num_epochs):
            for i, (index, image, label) in enumerate(self.dataloader):
                print(image.size())
                image = torch.autograd.Variable(image, requires_grad=False).to(self.device)
                label = torch.autograd.Variable(label, requires_grad=False).to(self.device).squeeze(1)

                prediction = self.model(image)  # ["out"]
                self.model.zero_grad()
                loss = self.criterion(prediction, label.long())
                loss.backward()
                self.optimizer.step()
                prediction = torch.argmax(prediction[0], 0).cpu().detach().numpy()

                if e > 8:
                    self.show(image, label, prediction)
                self.print_status(e, i, loss)
                if (e % self.config["utils"]["save_frequency"]) == 0:
                    pass # self.save_model(e)
                if (e + 1) == num_epochs:
                    self.save_model("final")

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
            loss.data.cpu().numpy()
        ))

    def save_model(self, name):
        if type(name) is not str:
            name = str(name)
        print("Saving model {}.torch".format(name))
        torch.save(self.model.state_dict(), "high_{}.torch".format(name))
