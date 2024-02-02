import os
import sys
import argparse

sys.path.append(os.getcwd())
from src.config.methods_configuration import MethodsConfiguration

from src.services.camera.depthcamera.realsense_d435i import RealsenseD435I
from src.methods.neuralnetworks.segmentation.u_mask_rcnn.u_mask_rcnn import UMaskRCNN
from src.methods.neuralnetworks.segmentation.yolov7.yolo_v7 import Yolov7
from src.methods.neuralnetworks.poseestimation.augmentedautoencoder.augmented_autoencoder import AugmentedAutoencoder

from src.services.thread.methods.methods_pipeline_thread import MethodsPipelineThread


def pipeline_mask_rcnn():
    real_sense_d4351 = RealsenseD435I((MethodsConfiguration.WIDTH, MethodsConfiguration.HEIGHT), False)

    u_mask_rcnn = UMaskRCNN("src/config/mask_rcnn.yml")
    augmented_autoencoder = AugmentedAutoencoder(MethodsConfiguration.PATH, False)

    methods_pipeline_thread = MethodsPipelineThread(real_sense_d4351, u_mask_rcnn, augmented_autoencoder)
    methods_pipeline_thread.run()


def pipeline_yolo_v7():
    real_sense_d4351 = RealsenseD435I((MethodsConfiguration.WIDTH, MethodsConfiguration.HEIGHT), False)

    yolo_v7 = Yolov7(MethodsConfiguration.PATH)
    augmented_autoencoder = AugmentedAutoencoder(MethodsConfiguration.PATH, False)

    methods_pipeline_thread = MethodsPipelineThread(real_sense_d4351, yolo_v7, augmented_autoencoder)
    methods_pipeline_thread.run()


def main(args):
    if args.segmentation_method == "mask_rcnn":
        pipeline_mask_rcnn()
    elif args.segmentation_method == "yolo_v7":
        pipeline_yolo_v7()
    else:
        exit(-1)
    try:
        pass
    except Exception as e:
        print(e)
        exit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sm", "--segmentation_method", required=False, help="mask_rcnn, yolo_v7", default="mask_rcnn")
    args = parser.parse_args()
    main(args)


