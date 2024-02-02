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


class TestFasterRCNN:
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

    def define_transform(self):
        self.transform_image = Compose([
            ToPILImage(),
            #Resize(self.config["transform"]["image_size"]),
            ToTensor(),
            #RandomErasing(p=0.8, scale=(0.1, 0.1), ratio=(0.1, 0.1))
        ])

    def load_model(self, model):
        self.model = model.to(self.device)
        model.load_state_dict(torch.load(self.config["general"]["model_path"]))
        self.model.eval()

    def predict(self, image_path):
        image = read_image(image_path)
        image = self.transform_image(image)
        image = image.to(self.device)
        with torch.no_grad():
            prediction = self.model([image])

        boxes = prediction[0]["boxes"]
        scores = prediction[0]["scores"]
        bb_indexing = scores > 0.8 # nms(boxes, scores, 0.9)
        boxes = boxes[bb_indexing]
        scores = scores[bb_indexing]
        non_max_suppression = nms(boxes, scores, 0.1)
        boxes = boxes[non_max_suppression]
        scores = scores[non_max_suppression]
        labels = [str(i.item()) for i in prediction[0]["labels"][non_max_suppression]]

        # bounding_box_image = draw_bounding_boxes((image*255).type(torch.uint8), boxes, colors=self.colors[int(labels[0])], labels=labels)
        #image = (image*255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        bounding_box_image = (image*255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0)
        bounding_box_image = bounding_box_image.copy()
        for bb, l, s in zip(boxes, labels, scores):
            text = "{}: {:.3f}".format(l, s)
            cv2.putText(
                bounding_box_image, text,
                (int(bb[0]), int(bb[1]) - 5), cv2.FONT_ITALIC, .5, self.colors[int(l)], 2
            )
            cv2.rectangle(bounding_box_image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), self.colors[int(l)], 1)
        #cv2.imshow("image", cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB))
        #cv2.waitKey()

        return (image*255).type(torch.uint8), bounding_box_image, boxes, [int(l) for l in labels], scores.cpu().numpy()

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
