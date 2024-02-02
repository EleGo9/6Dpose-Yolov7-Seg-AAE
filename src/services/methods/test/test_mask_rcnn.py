import time

import yaml

import numpy as np
import cv2
import torch
from torch.nn import *
from torchvision.transforms import *
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes, nms
from torchvision.transforms.functional import *


class TestMaskRCNN:
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

        self.colors = [(66, 66, 66), (255, 152, 0), (33, 150, 243), (244, 67, 54)]

        self.initialize()

    def initialize(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)
        print(self.config)

    def predict(self, image_path, threshold=0.8):
        image = read_image(image_path)
        image = self.transform_image(image)
        image = image.to(self.device)
        with torch.no_grad():
            prediction = self.model([image])

        labels = prediction[0]["labels"]
        scores = prediction[0]["scores"]
        boxes = prediction[0]["boxes"]
        masks = prediction[0]["masks"]

        threshold_indexing = scores > threshold
        labels = labels[threshold_indexing]
        scores = scores[threshold_indexing]
        boxes = boxes[threshold_indexing]
        masks = masks[threshold_indexing]

        return labels, scores.cpu().numpy(), boxes.cpu().numpy(), masks.cpu().numpy()

        # non_max_suppression = nms(boxes, scores, 0.1)
        # boxes = boxes[non_max_suppression]
        # scores = scores[non_max_suppression]
        # labels = [str(i.item()) for i in prediction[0]["labels"]]

        # bounding_box_image = draw_bounding_boxes((image*255).type(torch.uint8), boxes, colors=self.colors[int(labels[0])], labels=labels)
        # image = (image*255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        '''bounding_box_image = (image*255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0)
        bounding_box_image = bounding_box_image.copy()
        for bb, l, s in zip(boxes, labels, scores):
            text = "{}: {:.3f}".format(l, s)
            cv2.putText(
                bounding_box_image, text,
                (int(bb[0]), int(bb[1]) - 5), cv2.FONT_ITALIC, .5, self.colors[int(l)], 2
            )
            cv2.rectangle(bounding_box_image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), self.colors[int(l)], 1)

        masks = torch.argmax(prediction[0]["masks"], 1).detach().cpu().squeeze(0)
        masks = masks.detach().cpu().type(torch.uint8).numpy()
        masks = np.stack((masks, masks, masks))
        for i, channel in enumerate(masks):
            channel[channel == 1] = self.colors[1][i]
            channel[channel == 2] = self.colors[2][i]
            channel[channel == 3] = self.colors[3][i]
        masks = masks.transpose((1, 2, 0))

        image = (image.squeeze(0) * 255).cpu().detach().type(torch.uint8).numpy().transpose(1, 2, 0)

        cv2.imshow("image", cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey()'''

        threshold = 0.8
        pred_length = len(prediction[0]["scores"][prediction[0]["scores"] > threshold])
        if pred_length == 0 or pred_length > 9:
            #print(pred_length)
            return

        l = [str(i.item()) for i in prediction[0]["labels"][prediction[0]["scores"] > threshold]]
        bb = prediction[0]["boxes"][prediction[0]["scores"] > threshold]
        m = prediction[0]["masks"][prediction[0]["scores"] > threshold]
        # print(prediction[0]["scores"][prediction[0]["scores"] > threshold])
        m = m.squeeze(1)
        m = (m*255).type(torch.uint8)
        m[m != 0] = 255
        m = m.type(torch.bool)
        #m = torch.argmax(prediction[0]["masks"], dim=0).type(torch.bool)
        '''m = m.cpu().numpy().transpose(1, 2, 0)
        cv2.imshow("image", cv2.cvtColor(m, cv2.COLOR_BGR2RGB))
        cv2.waitKey()'''

        '''for im in m:
            print(im.unique())
            im[im != 0] = 255
            im = im.unsqueeze(0).type(torch.uint8).cpu().numpy().transpose(1, 2, 0)
            cv2.imshow("image", im)
            cv2.waitKey()'''

        colors = ["red", "green", "blue", "purple", "orange", "black", "yellow", "white", "aqua"]
        bounding_box = draw_bounding_boxes((image * 255).type(torch.uint8), bb, colors=colors[:m.size()[0]], labels=l)
        im = draw_segmentation_masks(bounding_box, m, alpha=0.3, colors=colors[:m.size()[0]])
        im = im.cpu().numpy().transpose(1, 2, 0)
        cv2.imshow("image", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def define_transform(self):
        self.transform_image = Compose([
            ToPILImage(),
            #Resize(self.config["transform"]["image_size"]),
            ToTensor(),
            #RandomErasing(p=0.8, scale=(0.1, 0.1), ratio=(0.1, 0.1))
        ])

    def load_model(self, model):
        self.model = model.to(self.device)
        print(self.config["general"]["model_path"])
        model.load_state_dict(torch.load(self.config["general"]["model_path"]))
        self.model.eval()

    def show(self, image, labels, scores, boxes, masks):
        image = (image * 255).type(torch.uint8)
        scores = torch.from_numpy(scores)
        boxes = torch.from_numpy(boxes)
        masks = torch.from_numpy(masks)

        masks = masks.squeeze(1)
        masks = (masks * 255).type(torch.uint8)
        masks[masks != 0] = 255
        masks = masks.type(torch.bool)

        labels = [str(l.item()) for l in labels]

        colors = ["red", "green", "blue", "purple", "orange", "black", "yellow", "white", "aqua"]
        bounding_box = draw_bounding_boxes(image, boxes, colors=colors[:masks.size()[0]], labels=labels)
        masked_image = draw_segmentation_masks(bounding_box, masks, alpha=0.3, colors=colors[:masks.size()[0]])
        masked_image = masked_image.cpu().numpy().transpose(1, 2, 0)
        cv2.imshow("image", cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey()
        return masked_image
