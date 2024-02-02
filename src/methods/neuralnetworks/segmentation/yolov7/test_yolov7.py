import os
import sys
import cv2

sys.path.append(os.getcwd())

from yolo_v7 import Yolov7
from src.config.methods_configuration import MethodsConfiguration


rgb_image = cv2.imread("/home/hakertz-test/Scaricati/new_nic_DS_examples/exp_0_5_5_5_5_pic.png")

yolo_v7 = Yolov7(MethodsConfiguration.PATH)
labels, scores, boxes, masks = yolo_v7.segment(rgb_image)
segmentation_image = yolo_v7.draw(rgb_image, labels, scores, boxes, masks)

print(len(scores))
for l, b, s in zip(labels, boxes, scores):
    print(l, [b[0]/640, b[1]/480, b[2]/640, b[3]/480], s)

for s in scores[::-1]:
    print("{:.6f}".format(s))

cv2.imshow("segmentation", segmentation_image)
cv2.waitKey()
