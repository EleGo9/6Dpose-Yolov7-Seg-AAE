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
import glob
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov7.det.models.common import DetectMultiBackend
from yolov7.det.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov7.det.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov7.det.utils.plots import Annotator, colors, save_one_box
# from yolov7.det.utils.segment.general import process_mask, scale_masks
# from yolov7.seg.utils.segment.plots import plot_masks
from yolov7.det.utils.torch_utils import select_device, smart_inference_mode

COLORS = np.random.randint(0, 255, size=(len([0,1,2,3,4]), 3), dtype="uint8")

class YoloV7det:
    def __init__(self, test_configpath, debug_vis):
        self.debug_vis = debug_vis
        test_args = configparser.ConfigParser()
        test_args.read(test_configpath)

        self.det_threshold = eval(test_args.get('DETECTOR', 'det_threshold'))
        self.iou_threshold = eval(test_args.get('DETECTOR', 'iou_threshold'))
        self.max_detections = eval(test_args.get('DETECTOR', 'max_detections'))
        self.nms_threshold = eval(test_args.get('DETECTOR', 'nms_threshold'))
        weights_path = str(test_args.get('DETECTOR', 'detector_model_path'))
        cfg_file = str(test_args.get('DETECTOR', 'detector_config_path'))
        data_path = str(test_args.get('DETECTOR', 'data_path'))
        self.device = str(test_args.get('DETECTOR', 'device'))
        # Write your path if visualize True
        # project = ROOT / 'runs/predict-seg'
        self.name = 'exp',  # save results to project/name
        self.exist_ok = False,  # existing project/name ok, do not increment
        self.save_txt = False,  # save results to *.txt
        self.save_conf = False,  # save confidences in --save-txt labels
        self.save_crop = False,  # save cropped prediction boxes
        self.line_thickness = 1
        self.hide_labels = False

        weights = weights_path
        cfg = cfg_file
        print('--------------------------------------')
        print('cfg', cfg)
        print('weights', weights)
        print('--------------------------------------')

        # Directories
        self.save_dir = '/home/elena/repos/6Dpose-Yolov7-Seg-AAE'
        self.class_ids = [0, 1, 2, 3, 4, 5]
        #self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        #(self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir


        # source = str(source)
        # save_img = not nosave and not source.endswith('.txt')  # save inference images
        # is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        # is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        # webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        # if is_url and is_file:
        #     source = check_file(source)  # download

        # Load model
        #device = select_device(device)
        self.net = DetectMultiBackend(weights, device=self.device, data=data_path) # device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.net.stride, self.net.names, self.net.pt
        self.bs = 1  # batch_size

        # imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        # if webcam:
        #     view_img = check_imshow()
        #     cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        #     bs = len(dataset)  # batch_size
        # else:
        #     dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        #     bs = 1  # batch_size
        # vid_path, vid_writer = [None] * bs, [None] * bs



    # def segment(self, image, im0, visualize=False):
    #     (W, H) = image.shape[1:]
    #     self.imgsz = check_img_size(H, s=self.stride)  # check image size
    #
    #     # Dataloader
    #     # if webcam:
    #     #     view_img = check_imshow()
    #     #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    #     #     bs = len(dataset)  # batch_size
    #     # else:
    #     #dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
    #     # vid_path, vid_writer = [None] * bs, [None] * bs
    #
    #     # Run inference
    #     # self.net.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup
    #     seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    #     #for path, im, im0s, vid_cap, s in dataset:
    #     with dt[0]:
    #         im = torch.from_numpy(image).to(self.device)
    #         im = im.half() if self.net.fp16 else im.float()  # uint8 to fp16/32
    #         im /= 255  # 0 - 255 to 0.0 - 1.0
    #         if len(im.shape) == 3:
    #             im = im[None]  # expand for batch dim
    #
    #     # Inference
    #     start_time = time.time()
    #     with dt[1]:
    #         # print('save_dir', self.save_dir)
    #         visualize = increment_path(self.save_dir / 'image', mkdir=True) if visualize else False
    #         # pred, out = self.net(im, augment=False, visualize=visualize)
    #         pred = self.net(im, augment=False, visualize=visualize)
    #         # self.proto = out[1]
    #     end_time = time.time()
    #
    #     # NMS
    #     start_time_post = time.time()
    #     with dt[2]:
    #         pred = non_max_suppression(pred, conf_thres=self.det_threshold, iou_thres=self.iou_threshold, classes=None, agnostic=False, max_det=self.max_detections)
    #         #pred = non_max_suppression(pred, conf_thres=self.det_threshold, iou_thres=self.iou_threshold, classes=None, agnostic=False, max_det=self.max_detections, nm=32)
    #
    #     # Process predictions
    #     self.boxes = []
    #     for i, det in enumerate(pred):  # per image
    #         # annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
    #         if len(det):
    #             # Rescale boxes from img_size to im0 size
    #             det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
    #
    #             # self.masks = process_mask(self.proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
    #
    #             '''Uncomment if you want to print the masks: '''
    #             # masks2 = np.swapaxes(masks, 0, 2)
    #             # masks2 = np.swapaxes(masks2, 0, 1 )
    #             # print('masks', masks2.shape)
    #             # #masks2 = np.swapaxes(masks2, 0, 2)
    #             # masks2 = masks2.detach().cpu().numpy()
    #             # cv2.imshow('mask', masks2[:,:,0])
    #             # cv2.waitKey(0)
    #
    #
    #             # Rescale boxes from img_size to im0 size (in this case they are the same)
    #             # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
    #
    #             # det_converted = det[:, :4].detach().cpu().numpy()
    #             # [self.boxes.append(det_converted[s,:]) for s in range(len(det))]
    #             #
    #             # # Print results
    #             # for c in det[:, 5].unique():
    #             #     n = (det[:, 5] == c).sum()  # detections per class
    #             #         # s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
    #             #
    #             # # Mask plotting ----------------------------------------------------------------------------------------
    #             # mcolors = [colors(int(cls), True) for cls in det[:, 5]]
    #             # self.im_masks = plot_masks(im[i], self.masks, mcolors)
    #             # annotator.im = scale_masks(im.shape[2:], self.im_masks, im0.shape)  # scale to original h, w
    #             # Mask plotting ---------------------------------------------------------------------------------------
    #     end_time_post = time.time()
    #     print("YOLO Inference Time: {:.3f} s".format(end_time - start_time))
    #     print("YOLO Post-processing Time: {:.3f} s".format(end_time_post - start_time_post))
    #     self.labels = []
    #     self.scores = []
    #     for det in pred:
    #         [self.labels.append(int(det[k, 5])) for k in range(len(det))]
    #         [self.scores.append(det[l, 4]) for l in range(len(det))]
    #
    #     return self.labels, self.scores, self.boxes

    def detect(self, image, im0, visualize=False):
        self.boxes = list()
        self.confidences = list()
        self.class_ids = list()
        # (H, W) = image.shape[:2]
        (W, H) = image.shape[1:]
        self.imgsz = check_img_size(H, s=self.stride)  # check image size

        # Run inference
        # self.net.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        # for path, im, im0s, vid_cap, s in dataset:
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
            visualize = increment_path(self.save_dir / 'image', mkdir=True) if visualize else False
            pred = self.net(im, augment=False, visualize=visualize)
        end_time = time.time()

        # NMS
        start_time_post = time.time()
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres=self.det_threshold, iou_thres=self.iou_threshold, classes=None,
                                       agnostic=False, multi_label=False, labels=(), max_det=self.max_detections)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            # imc = im0.copy() if self.save_crop else im0  # for save_crop
            annotator = Annotator(np.ascontiguousarray(im0), line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                det_converted = det[:, :4].detach().cpu().numpy()
                [self.boxes.append(det_converted[s, :]) for s in range(len(det))]

                # Write results
                # for *xyxy, conf, cls in reversed(det):
                #     # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / self.gn).view(-1).tolist()  # normalized xywh
                #     print(f'Buonding box in the xywh format {xywh}')
                #         # line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                #         # with open(f'{txt_path}.txt', 'a') as f:
                #         #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                for *xyxy, conf, cls in reversed(det):
                    if self.save_crop:  # Add bbox to image
                        # mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else f'{self.names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

                end_time_post = time.time()
                print("YOLO Inference Time: {:.3f} s".format(end_time - start_time))
                print("YOLO Post-processing Time: {:.3f} s".format(end_time_post - start_time_post))
                self.labels = []
                self.scores = []
                for det in pred:
                    [self.labels.append(int(det[k, 5])) for k in range(len(det))]
                    [self.scores.append(det[l, 4]) for l in range(len(det))]

                return self.labels, self.scores, self.boxes

        # for i, det in enumerate(pred):  # per image
        #     annotator = Annotator(np.ascontiguousarray(im0), line_width=self.line_thickness, example=str(self.names))
        #     for detection in det:
        #         detection = detection.detach().cpu().numpy()
        #         # scores = detection[4]
        #         class_id = detection[5]# np.argmax(scores)
        #         confidence = detection[4] # scores[class_id]
        #         if confidence > self.det_threshold:
        #             box = detection[0:4] * np.array([W, H, W, H])
        #             (center_x, center_y, width, height) = box.astype("int")
        #             x = int(center_x - (width / 2))
        #             y = int(center_y - (height / 2))
        #
        #             self.boxes.append([x, y, int(width), int(height)])
        #             self.confidences.append(float(confidence))
        #             self.class_ids.append(class_id)
            # if len(det):
            #     # self.masks = process_mask(self.proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            #
            #     '''Uncomment if you want to print the masks: '''
            #     # masks2 = np.swapaxes(masks, 0, 2)
            #     # masks2 = np.swapaxes(masks2, 0, 1 )
            #     # print('masks', masks2.shape)
            #     # #masks2 = np.swapaxes(masks2, 0, 2)
            #     # masks2 = masks2.detach().cpu().numpy()
            #     # cv2.imshow('mask', masks2[:,:,0])
            #     # cv2.waitKey(0)
            #
            #     # Rescale boxes from img_size to im0 size (in this case they are the same)
            #     det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            #
            #     det_converted = det[:, :4].detach().cpu().numpy()
            #
            #     box = det_converted[:, 0:4] * np.array([W, H, W, H])
            #     (center_x, center_y, width, height) = box[0].astype("int")
            #     x = int(center_x - (width / 2))
            #     y = int(center_y - (height / 2))
            #     self.boxes.append([x, y, int(width), int(height)])
            #
            #     [self.boxes.append(det_converted[s, :]) for s in range(len(det))]


        # for output in layer_outs:
        #     for detection in output:
        #         scores = detection[5:]
        #         class_id = np.argmax(scores)
        #         confidence = scores[class_id]
        #         if confidence > self.det_threshold:
        #             box = detection[0:4] * np.array([W, H, W, H])
        #             (center_x, center_y, width, height) = box.astype("int")
        #             x = int(center_x - (width / 2))
        #             y = int(center_y - (height / 2))
        #
        #             self.boxes.append([x, y, int(width), int(height)])
        #             self.confidences.append(float(confidence))
        #             self.class_ids.append(class_id)
        end_time_post = time.time()

        print("YOLO Inference Time: {:.3f} s".format(end_time - start_time))
        print("YOLO Post-processing Time: {:.3f} s".format(end_time_post - start_time_post))
        self.labels = []
        self.scores = []
        for det in pred:
            [self.labels.append(int(det[k, 5])) for k in range(len(det))]
            [self.scores.append(det[l, 4]) for l in range(len(det))]

        return self.labels, self.scores, self.boxes

    def draw(self, image):
        start_time = time.time()
        im = image.copy()
        if len(self.boxes) > 0:
            for i in range(len(self.boxes)):
                # if i==3:
                # (x, y) = (int(self.boxes[i][0]), int(self.boxes[i][1]))
                # (w, h) = (int(self.boxes[i][2]), int(self.boxes[i][3]))
                (x1, y1) = (int(self.boxes[i][0]), int(self.boxes[i][1]))
                (x2, y2) = (int(self.boxes[i][2]), int(self.boxes[i][3]))
                try:
                    color = [int(c) for c in COLORS[self.class_ids[i]]]
                except:
                    color = [152, 223, 81]
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
                # cv2.rectangle(im, (x, y), (w, h), color, 2)
                text = "{}: {:.4f}".format(self.labels[i], self.scores[i])
                cv2.putText(
                    im, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                # cv2.putText(
                #     im, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                # )

                # cv2.imshow('detection', cv2.resize(image/255.,(256,256)))
                # cv2.waitKey(0)

                if self.debug_vis:
                    cv2.imshow('detection', im)
                    cv2.waitKey(0)
        else:
            print("no results to show")
        end_time = time.time()
        print("YOLO Drawing Time: {:.3f} s".format(end_time - start_time))
        return im
