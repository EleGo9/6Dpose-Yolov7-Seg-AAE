import os
import sys
import argparse
from glob import glob

import torch

sys.path.append(os.getcwd())
from src.services.camera.depthcamera.realsense_d435i import RealsenseD435I
from src.methods.model.faster_rcnn import FasterRCNNModel
from src.services.methods.test.test_faster_rcnn import TestFasterRCNN

from src.augmentedautoencoder import AugmentedAutoencoder


def test_dir_image():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = FasterRCNNModel(num_classes=4)
    model = model.return_model()

    test = TestFasterRCNN("src/config/test_faster_rcnn.yml", device)
    test.define_transform()
    test.load_model(model)

    from statistics import mean, variance
    confidence_scores_obj_1 = []
    confidence_scores_obj_2 = []
    confidence_scores_obj_3 = []

    for image_path in glob("dataset/test/noise/all/rgb/*.png"):
        image, bb, boxes, labels, scores = test.predict(image_path)
        print(labels)
        print(scores)
        for label, score in zip(labels, scores):
            if label == 1:
                confidence_scores_obj_1.append(score.item())
            elif label == 2:
                confidence_scores_obj_2.append(score.item())
            elif label == 3:
                confidence_scores_obj_3.append(score.item())

    print(len(confidence_scores_obj_1), len(confidence_scores_obj_2), len(confidence_scores_obj_1))

    print(mean(confidence_scores_obj_1), sum(confidence_scores_obj_1) / len(confidence_scores_obj_1))
    print(variance(confidence_scores_obj_1))

    print(mean(confidence_scores_obj_2), sum(confidence_scores_obj_2) / len(confidence_scores_obj_2))
    print(variance(confidence_scores_obj_2))

    print(mean(confidence_scores_obj_3), sum(confidence_scores_obj_3) / len(confidence_scores_obj_3))
    print(variance(confidence_scores_obj_3))
    print("Test finished")


def test_camera_image():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    camera = RealsenseD435I((640, 480), False)
    model = FasterRCNNModel(num_classes=3)
    model = model.return_model()

    test = TestFasterRCNN("src/config/test_faster_rcnn.yml", device)
    test.define_transform()
    test.load_model(model)

    while True:
        image_path = "dataset/test/rgb/camera.png"
        camera.photo(image_path)
        test.predict(image_path)


def test_aae():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #camera = RealsenseD435I((640, 480), False)
    model = CustomModel(num_channels=3, num_classes=2)
    # model = UNet(num_channels=3, num_classes=2, bilinear=False)
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    test_configpath = os.path.join(workspace_path, "cfg_eval/cem_obj_001.cfg")
    aae = AugmentedAutoencoder(test_configpath, False, False)

    test = Test("src/config/test.yml", device)
    test.define_transform()
    test.load_model(model)

    for image_path in glob("dataset/test/rgb/black/*.png"):
        import numpy as np
        import cv2
        # image_path = "dataset/test/rgb/camera.png"
        # image = camera.photo(image_path)
        #image = cv2.imread(image_path)
        masks, bounding_boxes, masked_image = test.predict(image_path)
        bounding_boxes = xyxy2xywh(bounding_boxes.squeeze(0))

        aae_boxes, scores, labels = aae.process_detection_output(480, 640, [bounding_boxes], [0], [0])
        all_pose_estimates, all_class_idcs, _ = aae.process_pose(aae_boxes, [0], masked_image)
        pose_6d = aae.draw(masked_image, all_pose_estimates, all_class_idcs, aae_boxes, scores, labels, [])

        (x1, y1) = (int(bounding_boxes[0]), int(bounding_boxes[1]))
        (x2, y2) = (int(bounding_boxes[2]), int(bounding_boxes[3]))
        cv2.rectangle(pose_6d, (x1, y1), (x2, y2), [0, 0, 255])

        full_show = np.concatenate((masked_image, pose_6d), axis=1)
        cv2.imshow('6D pose estimation', full_show)
        cv2.waitKey(10)


def xyxy2xywh(xyxy):
    (x1, y1) = (int(xyxy[0]), int(xyxy[1]))
    (x2, y2) = (int(xyxy[2]), int(xyxy[3]))
    w = int(x2-x1)
    h = int(y2-y1)
    return [x1, y1, w, h]


def main():
    try:
        if args.source == "directory":
            test_dir_image()
        elif args.source == "camera":
            test_camera_image()
        elif args.source == "aae":
            test_aae()
        else:
            print("Image source not correct")
            exit(-1)
    except Exception as e:
        sys.stderr.write(str(e) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", "--source", required=False, help="Image source", default="directory")
    args = parser.parse_args()
    main()
