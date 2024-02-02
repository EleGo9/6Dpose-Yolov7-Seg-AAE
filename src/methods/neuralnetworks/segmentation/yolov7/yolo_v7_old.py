import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import configparser
import torch
import torch.backends.cudnn as cudnn
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov7.seg.models.common import DetectMultiBackend
from yolov7.seg.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov7.seg.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov7.seg.utils.plots import Annotator, colors, save_one_box
from yolov7.seg.utils.segment.general import process_mask, scale_masks
from yolov7.seg.utils.segment.plots import plot_masks
from yolov7.seg.utils.torch_utils import select_device, smart_inference_mode

from src.methods.neuralnetworks.segmentation.interface_segmentation import ISegmentation

COLORS = np.random.randint(0, 255, size=(len([0,1,2,3,4]), 3), dtype="uint8")


class Yolov7(ISegmentation):
    def __init__(self, config_path, debug_vis=False):
        self.debug_vis = debug_vis
        test_args = configparser.ConfigParser()
        test_args.read(config_path)

        self.det_threshold = eval(test_args.get('DETECTOR', 'det_threshold'))
        self.max_detections = eval(test_args.get('DETECTOR', 'max_detections'))
        self.nms_threshold = eval(test_args.get('DETECTOR', 'nms_threshold'))
        weights_path = str(test_args.get('DETECTOR', 'detector_model_path'))
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
        self.net = DetectMultiBackend(weights, device=self.device, data=data_path) # device=device, dnn=dnn, data=data, fp16=half)
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

    def segment(self, image, im0, visualize=False):
        (W, H) = image.shape[1:]
        self.imgsz = check_img_size(H, s=self.stride)  # check image size

        # Dataloader
        # if webcam:
        #     view_img = check_imshow()
        #     cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        #     bs = len(dataset)  # batch_size
        # else:
        #dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        # vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        # self.net.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        #for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(image).to(self.device)
            im = im.half() if self.net.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        start_time = time.time()
        with dt[1]:
            # print('save_dir', self.save_dir)
            # visualize = increment_path(self.save_dir / 'image', mkdir=True) if visualize else False
            pred, out = self.net(im, augment=False, visualize=visualize)
            self.proto = out[1]
        end_time = time.time()

        # NMS
        start_time_post = time.time()
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres=self.nms_threshold, iou_thres=self.det_threshold, classes=None, agnostic=False, max_det=self.max_detections, nm=32)

        # Process predictions
        self.boxes = []
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(np.ascontiguousarray(im0), line_width=self.line_thickness, example=str(self.names))
            if len(det):
                self.masks = process_mask(self.proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                '''Uncomment if you want to print the masks: '''
                # masks2 = np.swapaxes(masks, 0, 2)
                # masks2 = np.swapaxes(masks2, 0, 1 )
                # print('masks', masks2.shape)
                # #masks2 = np.swapaxes(masks2, 0, 2)
                # masks2 = masks2.detach().cpu().numpy()
                # cv2.imshow('mask', masks2[:,:,0])
                # cv2.waitKey(0)


                # Rescale boxes from img_size to im0 size (in this case they are the same)
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                det_converted = det[:, :4].detach().cpu().numpy()
                [self.boxes.append(det_converted[s,:]) for s in range(len(det))]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                        # s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                self.im_masks = plot_masks(im[i], self.masks, mcolors)
                annotator.im = scale_masks(im.shape[2:], self.im_masks, im0.shape)  # scale to original h, w
                # Mask plotting ---------------------------------------------------------------------------------------
        end_time_post = time.time()
        #print("YOLO Inference Time: {:.3f} s".format(end_time - start_time))
        #print("YOLO Post-processing Time: {:.3f} s".format(end_time_post - start_time_post))
        self.labels = []
        self.scores = []
        for det in pred:
            [self.labels.append(int(det[k, 5])) for k in range(len(det))]
            [self.scores.append(det[l, 4]) for l in range(len(det))]

        for i in range(len(self.labels)):
            self.labels[i] += 1

        return self.labels, self.scores, self.boxes, self.masks, self.im_masks

    def draw(self, image):
        start_time = time.time()
        im = image.copy()
        if len(self.boxes) > 0:
            for i in range(len(self.boxes)):
                # if i==3:
                (x1, y1) = (int(self.boxes[i][0]), int(self.boxes[i][1]))
                (x2, y2) = (int(self.boxes[i][2]), int(self.boxes[i][3]))
                try:
                    color = [int(c) for c in COLORS[self.class_ids[i]]]
                except:
                    color = [152, 223, 81]
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
                text = "{}: {:.4f}".format(self.labels[i], self.scores[i])
                cv2.putText(
                    im, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

                # cv2.imshow('detection', cv2.resize(image/255.,(256,256)))
                if self.debug_vis:
                    cv2.imshow('detection', im)
                    cv2.waitKey(0)
        else:
            print("no results to show")
        end_time = time.time()
        print("YOLO Drawing Time: {:.3f} s".format(end_time - start_time))
        return im

    def draw(self, image, labels, scores, boxes, masks):
        from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
        image = torch.from_numpy(image.transpose(2, 0, 1)).type(torch.uint8)
        masks = masks.type(torch.bool)
        boxes = torch.from_numpy(np.array(boxes, dtype=np.float))
        labels = ["{}: {:.3f}".format(l, s) for l, s in zip(labels, scores)]
        print(boxes)

        colors = ["red", "green", "blue", "orange", "white", "purple", "yellow", "aqua", "black", "brown", "red", "green", "blue", "orange", "white", "purple", "yellow", "aqua", "black", "brown"]
        bounding_box = draw_bounding_boxes(image, boxes, colors=colors[:masks.size()[0]]) #, labels=labels)
        masked_image = draw_segmentation_masks(bounding_box, masks, alpha=0.3, colors=colors[:masks.size()[0]])
        masked_image = masked_image.cpu().numpy().transpose(1, 2, 0)
        for l, b in zip(labels, boxes):
            cv2.putText(masked_image, l,
                        (int(b[0]), int(b[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (30, 200, 255), 2)
        # cv2.imshow("image", cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        # cv2.waitKey()
        return masked_image