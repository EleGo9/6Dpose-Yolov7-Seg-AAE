import os
import sys
import argparse
from glob import glob

import torch
import numpy as np
import cv2

sys.path.append(os.getcwd())
from src.services.camera.depthcamera.realsense_d435i import RealsenseD435I
from src.methods.model.mask_rcnn import MaskRCNNModel, MaskRCNNPredictor
from src.services.methods.test.test_mask_rcnn import TestMaskRCNN

from src.methods.neuralnetworks.segmentation.u_mask_rcnn.u_mask_rcnn import UMaskRCNN
from src.methods.neuralnetworks.poseestimation.augmentedautoencoder.augmented_autoencoder import AugmentedAutoencoder


def test_dir_image():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MaskRCNNModel(num_classes=4, mask_predictor=MaskRCNNModel.EXTENDED_U_MASK_RCNN)
    model = model.return_model()

    test = TestMaskRCNN("src/config/test_mask_rcnn.yml", device)
    test.define_transform()
    test.load_model(model)

    '''from statistics import mean, variance
    confidence_scores_obj_1 = []
    confidence_scores_obj_2 = []
    confidence_scores_obj_3 = []'''

    '''for i in range(4500):
        image_path = "dataset/train/objects/all/mask/{:04d}.png".format(i)

        if not os.path.isfile(image_path):
            print(image_path)'''

    for image_path in glob("dataset/test/objects/all/test/rgb/*.png"):
        test.predict(image_path)

        '''print(labels)
        print(scores)
        for label, score in zip(labels, scores):
            if label == 1:
                confidence_scores_obj_1.append(score.item())
            elif label == 2:
                confidence_scores_obj_2.append(score.item())
            elif label == 3:
                confidence_scores_obj_3.append(score.item())'''

    '''print(len(confidence_scores_obj_1), len(confidence_scores_obj_2), len(confidence_scores_obj_1))

    print(mean(confidence_scores_obj_1), sum(confidence_scores_obj_1) / len(confidence_scores_obj_1))
    print(variance(confidence_scores_obj_1))

    print(mean(confidence_scores_obj_2), sum(confidence_scores_obj_2) / len(confidence_scores_obj_2))
    print(variance(confidence_scores_obj_2))

    print(mean(confidence_scores_obj_3), sum(confidence_scores_obj_3) / len(confidence_scores_obj_3))
    print(variance(confidence_scores_obj_3))'''
    print("Test finished")


def test_camera_image():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    camera = RealsenseD435I((640, 480), False)

    model = MaskRCNNModel(num_classes=4, mask_predictor=MaskRCNNModel.EXTENDED_U_MASK_RCNN)
    model = model.return_model()

    test = TestMaskRCNN("src/config/test_mask_rcnn.yml", device)
    test.define_transform()
    test.load_model(model)

    while True:
        image_path = "dataset/test/rgb/camera.png"
        camera.photo(image_path)
        test.predict(image_path)


def test_aae():
    #camera = RealsenseD435I((640, 480), False)

    u_mask_rcnn = UMaskRCNN("src/config/mask_rcnn.yml")

    workspace_path = os.environ.get("AE_WORKSPACE_PATH")
    test_configpath = os.path.join(workspace_path, "cfg_eval/cem_objs.cfg")
    print(test_configpath)
    aae = AugmentedAutoencoder(test_configpath, False, False)

    for image_path in glob("dataset/test/objects/all/test/rgb/*.png"):
        #image_path = "dataset/test/rgb/camera.png"
        #image = camera.photo(image_path)
        image = cv2.imread(image_path)

        labels, scores, boxes, masks = u_mask_rcnn.segment(image)
        for i in range(len(boxes)):
            boxes[i] = xyxy2xywh(boxes[i])

        masked_image = mask_image(image, masks)
        boxes, scores, labels, all_pose_estimates, all_class_idcs, all_cosine_similarity = aae.pose_estimation(masked_image, labels, boxes, scores)
        pose_6d, _ = aae.draw(image, all_pose_estimates, all_class_idcs, labels, boxes, scores, all_cosine_similarity)

        full_show = np.concatenate((image, pose_6d), axis=1)
        cv2.imshow("6D pose estimation", full_show)
        cv2.waitKey()

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
