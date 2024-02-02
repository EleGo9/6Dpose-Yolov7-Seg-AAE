import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import sys
import configparser
import torch
import torch.backends.cudnn as cudnn
import time
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.append("/home/hakertz-test/repos/6Dpose-Yolov7-Seg-AAE/yolov7/seg/")
from yolov7.seg.models.common import DetectMultiBackend
from yolov7.seg.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov7.seg.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
                                      colorstr, cv2,
                                      increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer,
                                      xyxy2xywh)
from yolov7.seg.utils.plots import Annotator, colors, save_one_box
from yolov7.seg.utils.segment.general import process_mask, scale_masks
from yolov7.seg.utils.segment.plots import plot_masks
from yolov7.seg.utils.torch_utils import select_device, smart_inference_mode

from src.methods.neuralnetworks.segmentation.interface_segmentation import ISegmentation

COLORS = np.random.randint(0, 255, size=(len([0, 1, 2, 3, 4]), 3), dtype="uint8")


class Yolov7(ISegmentation):
    def __init__(self, config_path, debug_vis=False):
        self.debug_vis = debug_vis
        test_args = configparser.ConfigParser()
        test_args.read(config_path)

        self.det_threshold = eval(test_args.get('DETECTOR', 'det_threshold'))
        self.nms_threshold = eval(test_args.get('DETECTOR', 'nms_threshold'))
        self.iou_threshold = eval(test_args.get('DETECTOR', 'iou_threshold'))
        self.max_detections = eval(test_args.get('DETECTOR', 'max_detections'))
        weights_path = str(test_args.get('DETECTOR', 'detector_model_path'))
        print(weights_path)
        cfg_file = str(test_args.get('DETECTOR', 'detector_config_path'))
        data_path = str(test_args.get('DETECTOR', 'data_path'))
        self.device = str(test_args.get('DETECTOR', 'device'))
        # Write your path if visualize True
        project = ROOT / 'runs/predict-seg'
        name = 'exp',  # save results to project/name
        exist_ok = False,  # existing project/name ok, do not increment
        save_txt = False,  # save results to *.txt
        save_conf = False,  # save confidences in --save-txt labels
        save_crop = False,  # save cropped prediction boxes
        self.line_thickness = 3

        weights = weights_path
        cfg = cfg_file

        self.class_ids = [0, 1, 2]
        self.net = DetectMultiBackend(weights, device=self.device,
                                      data=data_path)  # device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.net.stride, self.net.names, self.net.pt
        self.bs = 1
        self.imgsz = None
        self.proto = None
        self.boxes = None
        self.masks = None
        self.im_masks = None
        self.labels = None
        self.scores = None

    def initialize(self):
        pass

    def segment(self, image):
        #image = np.swapaxes(image, 0, 2)
        #image = np.swapaxes(image, 1, 2)
        image = np.ascontiguousarray(image.transpose((2, 0, 1))[::-1])
        im0 = image
        (W, H) = image.shape[1:]
        self.imgsz = check_img_size(H, s=self.stride)
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        with dt[0]:
            im = torch.from_numpy(image).to(self.device)
            im = im.half() if self.net.fp16 else im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        with dt[1]:
            pred, out = self.net(im, augment=False, visualize=False)
            self.proto = out[1]

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres=self.det_threshold, iou_thres=self.iou_threshold, classes=None,
                                       agnostic=False, max_det=self.max_detections, nm=32)

        self.boxes = []
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(np.ascontiguousarray(im0), line_width=self.line_thickness, example=str(self.names))
            if len(det):
                self.masks = process_mask(self.proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det_converted = det[:, :4].detach().cpu().numpy()
                [self.boxes.append(det_converted[s, :]) for s in range(len(det))]

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()

                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                self.im_masks = plot_masks(im[i], self.masks, mcolors)
                annotator.im = scale_masks(im.shape[2:], self.im_masks, im0.shape)  # scale to original h, w
        self.labels = []
        self.scores = []
        for det in pred:
            [self.labels.append(int(det[k, 5])) for k in range(len(det))]
            [self.scores.append(det[l, 4]) for l in range(len(det))]

        '''for i in range(len(self.labels)):
            self.labels[i] -= 1'''
        self.scores = [s.cpu().numpy() for s in self.scores]

        return self.labels, self.scores, self.boxes, self.masks #, self.im_masks

    def draw(self, image, labels, scores, boxes, masks):
        image = torch.from_numpy(image.transpose(2, 0, 1)).type(torch.uint8)
        masks = masks.type(torch.bool)
        boxes = torch.from_numpy(np.array(boxes, dtype=np.float))
        labels = ["{}: {:.3f}".format(l, s) for l, s in zip(labels, scores)]

        colors = ["red", "green", "blue", "orange", "white", "purple", "yellow", "aqua", "black", "brown", "red",
                  "green", "blue", "orange", "white", "purple", "yellow", "aqua", "black", "brown",
                  "red", "green", "blue", "orange", "white", "purple", "yellow", "aqua", "black", "brown", "red",
                  "green", "blue", "orange", "white", "purple", "yellow", "aqua", "black", "brown"
                  ]
        bounding_box = draw_bounding_boxes(image, boxes, colors=colors[:masks.size()[0]])  # , labels=labels)
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
        masks = masks.detach().cpu().numpy()
        unified_mask = masks[0]
        for i in range(1, masks.shape[0]):
            unified_mask = cv2.bitwise_or(unified_mask, masks[i], mask=None)
        unified_mask = unified_mask.astype("uint8")
        masked = cv2.bitwise_and(image, image, mask=unified_mask)
        return masked
