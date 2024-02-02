import yaml

import numpy as np
import cv2
import torch
from torch.nn import *
from torchvision.transforms import *
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes


class Test:
    def __init__(
            self,
            config_path: str,
            device: torch.device = torch.device("cuda:0")
    ):
        self.config_path = config_path
        self.device = device

        self.config = None
        self.transform_image = None
        self.model = None

        self.initialize()

    def initialize(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)
        print(self.config)

    def define_transform(self):
        self.transform_image = Compose([
            ToPILImage(),
            #Resize(self.config["transform"]["image_size"]),
            ToTensor(),
            RandomErasing(p=0.8, scale=(0.1, 0.1), ratio=(0.1, 0.1))
        ])

    def load_model(self, model: Module):
        self.model = model.to(self.device)
        model.load_state_dict(torch.load(self.config["general"]["model_path"]))
        self.model.eval()

    def predict(self, image_path):
        image = read_image(image_path)
        if image.size()[0] > 3:
            image = image[:3, :, :]
        image = self.transform_image(image)
        image = torch.autograd.Variable(image, requires_grad=False).to(self.device).unsqueeze(0)
        with torch.no_grad():
            masks = self.model(image) #["out"]

        '''resize = Resize((480, 640))
        image = resize(image)
        masks = resize(masks)'''
        masks = torch.argmax(masks[0], 0).cpu().detach()
        bounding_boxes = masks_to_boxes(masks.unsqueeze(0))
        masked_image = self.show(image, masks, bounding_boxes)

        masks = masks.numpy()
        bounding_boxes = bounding_boxes.numpy()

        return masks, bounding_boxes, masked_image

    def show(self, image, masks, bounding_boxes):
        show_interval = self.config["utils"]["show"]
        if show_interval is not None:
            image_show = (image[0]*255).cpu().detach().type(torch.uint8) #.numpy().transpose(1, 2, 0)
            masks_show = masks.cpu().detach().type(torch.bool)
            _masks_show = draw_segmentation_masks(image_show, masks_show, alpha=0.2, colors="red")
            bounding_boxes_show = draw_bounding_boxes(image_show, bounding_boxes, colors="red")

            image_show = image_show.numpy().transpose(1, 2, 0)
            bounding_boxes_show = bounding_boxes_show.numpy().transpose(1, 2, 0)
            masks_show = masks_show.numpy().astype(np.uint8) #.transpose(1, 2, 0)
            image_show = cv2.bitwise_not(image_show)
            sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            image_show = cv2.filter2D(image_show, -1, sharpen_filter)
            masked_image = cv2.bitwise_and(image_show, image_show, mask=masks_show)
            _masks_show = _masks_show.numpy().astype(np.uint8).transpose(1, 2, 0)
            full_show = np.concatenate((bounding_boxes_show, _masks_show), axis=1)
            cv2.imshow("image - prediction", cv2.cvtColor(full_show, cv2.COLOR_BGR2RGB))
            #cv2.waitKey(0)


            #cv2.imshow("image", cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
            #cv2.waitKey(0)
            return masked_image
