import yaml

import numpy as np
import cv2
import torch
from torch.nn import *
from torchvision.transforms import *
from torch.optim import *
from torch.optim.lr_scheduler import StepLR
from src.utils.vision.references.detection.engine import train_one_epoch, evaluate
from src.utils.vision.references.detection.utils import *

from src.methods.dataset.mask_rcnn_dataset import DatasetMaskRCNN, DatasetMaskRCNNEvaluation
from src.methods.dataloader.custom_dataloader import CustomDataloader


class TrainMaskRCNN:
    def __init__(self, config_path: str):
        self.config_path = config_path

        self.device = None
        self.config = None
        self.transform_image = None
        self.transform_mask = None
        self.train_dataset = None
        self.eval_dataset = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.model = None
        self.optimizer = None
        self.learning_rate = None

        self.initialize()

    def initialize(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)

        for k, v in self.config.items():
            print("{}: {}".format(k, v))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.train_dataset = DatasetMaskRCNN(
            self.config["general"]["train_path"],
            transform_image=self.transform_image,
            transform_mask=self.transform_mask
        )

        self.eval_dataset = DatasetMaskRCNNEvaluation(
            self.config["general"]["eval_path"],
            transform_image=self.transform_image,
            transform_mask=self.transform_mask
        )

    def define_dataloader(self):
        self.train_dataloader = CustomDataloader(
            dataset=self.train_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=self.config["hyper_parameters"]["shuffle"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            collate_fn=collate_fn
        )

        self.eval_dataloader = CustomDataloader(
            dataset=self.eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )

    def define_model(self, model, pre_trained: bool = False):
        self.model = model.to(self.device)
        if pre_trained:
            print("Loaded {}".format(self.config["general"]["weights"]))
            self.model.load_state_dict(torch.load(self.config["general"]["weights"]))
        self.model.train()

    def define_optimizer(self):
        self.optimizer = SGD(
            params=self.model.parameters(),
            lr=self.config["hyper_parameters"]["learning_rate"],
            momentum=self.config["hyper_parameters"]["momentum"],
            weight_decay=self.config["hyper_parameters"]["weight_decay"]
        )
        self.learning_rate = StepLR(
            self.optimizer,
            step_size=self.config["hyper_parameters"]["step_size"],
            gamma=self.config["hyper_parameters"]["gamma"]
        )

    def train(self):
        num_epochs = self.config["hyper_parameters"]["num_epochs"]
        for epoch in range(num_epochs):
            train_one_epoch(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.device,
                epoch,
                print_freq=10
            )
            self.learning_rate.step()

            evaluate(self.model, self.eval_dataloader, device=self.device)

            self.save_model()
        self.save_model()

    def save_model(self):
        name = self.config["general"]["save_model_path"]
        print("Saving model {}.torch".format(name))
        torch.save(self.model.state_dict(), "{}".format(name))
