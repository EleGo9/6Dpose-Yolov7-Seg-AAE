import os
import sys
import argparse
from glob import glob

import numpy as np
import torch
from torchvision.transforms import *

import cv2
from time import time

sys.path.append(os.getcwd())
from src.services.camera.depthcamera.realsense_d435i import RealsenseD435I
from src.methods.model.sg_det import SGDet
from src.methods.model.custom_model import CustomModel
from src.methods.model.unet import UNet
from src.services.methods.test.test_segmentation import TestSegmentation

from src.methods.model.faster_rcnn import FasterRCNNModel
from src.services.methods.test.test_faster_rcnn import TestFasterRCNN

from src.augmentedautoencoder import AugmentedAutoencoder

from src.methods.imageprocessing.pickingpoint.object_picking_point import ObjectPickingPoint

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms.functional import *
from torchvision.ops import *


def test_dir_image():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model = deeplabv3_resnet50(pretrained=True)
    #model.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    model = SGDet(num_channels=3, num_classes=4)
    #model = UNet(num_channels=3, num_classes=3)
    #model = CustomModel(num_channels=3, num_classes=3)

    test = TestSegmentation("src/config/test_segmentation.yml", device)
    test.define_transform()
    test.load_model(model)

    # model.export_onnx(image)

    for image_path in glob("dataset/test/noise/all/rgb/*.png"):
        test.predict(image_path)
    print("Test finished")


def test_det_seg():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #camera = RealsenseD435I((640, 480), False)

    detection = FasterRCNNModel(num_classes=4)
    detection = detection.return_model()
    test_detection = TestFasterRCNN("src/config/test_faster_rcnn.yml", device)
    test_detection.define_transform()
    test_detection.load_model(detection)

    segmentation = SGDet(num_channels=3, num_classes=4)
    segmentation_test = TestSegmentation("src/config/test_segmentation.yml", device)
    segmentation_test.define_transform()
    segmentation_test.load_model(segmentation)

    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    test_config_path = os.path.join(workspace_path, "cfg_eval/cem_objs.cfg")
    aae = AugmentedAutoencoder(test_config_path, False, False)

    '''from statistics import mean, variance
    confidence_scores_obj_1 = []
    confidence_scores_obj_2 = []
    confidence_scores_obj_3 = []'''
    for image_path in glob("dataset/test/noise/all/rgb/*.png"):
        #while True:
        #    image_path = "dataset/test/rgb/camera.png"
        #    camera.photo(image_path)

        start_time = time()
        image, bounding_box_image, boxes, labels, scores = test_detection.predict(image_path)
        print("detection time", time() - start_time)
        start_time = time()
        mask_image, class_mask = segmentation_test.predict(image_path)
        print("segmentation time", time() - start_time)

        if True:
            for box, label, in zip(boxes, labels):
                mask_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])][mask_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] != label] = 0

        image = image.cpu().numpy().transpose(1, 2, 0)
        filter_image = cv2.bitwise_not(image)
        sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        filter_image = cv2.filter2D(filter_image, -1, sharpen_filter)
        filter_image = cv2.bitwise_and(filter_image, filter_image, mask=mask_image)

        boxes = box_convert(boxes, "xyxy", "xywh")
        boxes = boxes.cpu().numpy()
        boxes, scores, labels = aae.process_detection_output(480, 640, boxes, scores, labels)
        all_pose_estimates, all_class_idcs, all_cs = aae.process_pose(boxes, labels, filter_image)
        pose_6d = aae.draw(image, all_pose_estimates, all_class_idcs, boxes, scores, labels, all_cs)

        print(class_mask.shape, filter_image.shape)
        show_up = np.concatenate((class_mask, filter_image), axis=1)
        show_down = np.concatenate((bounding_box_image, pose_6d), axis=1)
        full_image = np.concatenate((show_up, show_down), axis=0)
        #cv2.imwrite("src/image/custom_pipeline.png", cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB))
        #cv2.imshow("6d pose", cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB))
        #cv2.waitKey()

        '''for label, cs in zip(labels, all_cs):
            if label == 1:
                confidence_scores_obj_1.append(cs)
            elif label == 2:
                confidence_scores_obj_2.append(cs)
            elif label == 3:
                confidence_scores_obj_3.append(cs)'''

    '''print(len(confidence_scores_obj_1) + len(confidence_scores_obj_2) + len(confidence_scores_obj_1))

    print(mean(confidence_scores_obj_1), sum(confidence_scores_obj_1) / len(confidence_scores_obj_1))
    print(np.var(confidence_scores_obj_1))

    print(mean(confidence_scores_obj_2), sum(confidence_scores_obj_2) / len(confidence_scores_obj_2))
    print(np.var(confidence_scores_obj_2))

    print(mean(confidence_scores_obj_3), sum(confidence_scores_obj_3) / len(confidence_scores_obj_3))
    print(np.var(confidence_scores_obj_3))

    print("Test finished")'''


def test_camera_image():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    camera = RealsenseD435I((640, 480), False)

    detection = FasterRCNNModel(num_classes=3)
    detection = detection.return_model()
    test_detection = TestFasterRCNN("src/config/test_faster_rcnn.yml", device)
    test_detection.define_transform()
    test_detection.load_model(detection)

    segmentation = SGDet(num_channels=3, num_classes=3)
    segmentation_test = TestSegmentation("src/config/test_segmentation.yml", device)
    segmentation_test.define_transform()
    segmentation_test.load_model(segmentation)

    while True:
        image_path = "dataset/test/rgb/camera.png"
        camera.photo(image_path)
        test_detection.predict(image_path)
        segmentation_test.predict(image_path)


def test_aae():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #camera = RealsenseD435I((640, 480), False)
    model = SGDet(num_channels=3, num_classes=2)
    #model = CustomModel(num_channels=3, num_classes=2)
    # model = UNet(num_channels=3, num_classes=2, bilinear=False)
    # model_faster_rcnn = FasterRCNNModel(num_classes=3)
    # model_faster_rcnn = model_faster_rcnn.return_model()
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    test_configpath = os.path.join(workspace_path, "cfg_eval/cem_objs.cfg")
    aae = AugmentedAutoencoder(test_configpath, False, False)

    test = TestSegmentation("src/config/test_segmentation.yml", device)
    test.define_transform()
    test.load_model(model)
    '''test_faster_rcnn = TestFasterRCNN("src/config/test_faster_rcnn.yml", device)
    test_faster_rcnn.define_transform()
    test_faster_rcnn.load_model(model_faster_rcnn)'''

    for image_path in glob("dataset/test/rgb/black/*.png"):
        import numpy as np
        import cv2
        # image_path = "dataset/test/rgb/camera.png"
        # image = camera.photo(image_path)
        #image = cv2.imread(image_path)
        #test_faster_rcnn.predict(image_path)
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
        cv2.waitKey(0)


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
        elif args.source == "det_seg":
            test_det_seg()
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


'''transform = Compose([transforms.ToPILImage(), transforms.ToTensor()])
        filter_segment = transform(filter_image)
        crop_images = []
        for bb in boxes:
            crop_images.append(crop(filter_segment, int(bb[1]), int(bb[0]), int(bb[3]), int(bb[2])))

        objects_center = []
        for ci in crop_images:
            ci = ci.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            objects_center.append(list(picking_point.find(ci)))

        for oc, bb in zip(objects_center, boxes):
            oc[0] = oc[0] // 2 + int(bb[0])
            oc[1] = oc[1] // 2 + int(bb[1])

        euler_angles_z_rotations = []
        euler_angles_to_show = []
        rotations = []
        for pe in all_pose_estimates:
            rotation_vector, _ = cv2.Rodrigues(pe[:3, :3])
            points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
            axis_points, _ = cv2.projectPoints(
                points,
                np.array(rotation_vector.reshape(1, 3)[0]),
                np.array(pe[:3, 3]).T,
                aae.get_camK(), (0, 0, 0, 0)
            )

            b = axis_points[3].ravel()
            c = axis_points[1].ravel()
            a = [b[0] + 50, b[1]]
            rotations.append([
                tuple(axis_points[0].astype(int).ravel()),
                tuple(axis_points[1].astype(int).ravel()),
                tuple(axis_points[2].astype(int).ravel())
            ])

            z_angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))

            z_angle_to_show = z_angle
            if z_angle_to_show > 0:
                z_angle_to_show = np.pi / 2 - z_angle_to_show
            elif z_angle_to_show < 0:
                z_angle_to_show = -(np.pi / 2 + z_angle_to_show)
            euler_angles_to_show.append(z_angle_to_show)

            z_rotation = np.deg2rad(z_angle)
            if z_rotation > 0:
                z_rotation = np.pi / 2 - z_rotation
            elif z_rotation < 0:
                z_rotation = -(np.pi / 2 + z_rotation)
            euler_angles_z_rotations.append(z_rotation)

        for i, (oc, ozr, r) in enumerate(zip(objects_center, euler_angles_to_show, rotations)):
            div = 5
            pose_6d = cv2.line(pose_6d, oc, tuple([oc[0]+(r[1][0]-oc[0])//div, oc[1]+(r[1][1]-oc[1])//div]), (255, 0, 0), 2)
            pose_6d = cv2.line(pose_6d, oc, tuple([oc[0]+(r[0][0]-oc[0])//div, oc[1]+(r[0][1]-oc[1])//div]), (0, 255, 0), 2)
            pose_6d = cv2.line(pose_6d, oc, tuple([oc[0]+(r[2][0]-oc[0])//div, oc[1]+(r[2][1]-oc[1])//div]), (0, 0, 255), 2)

            cv2.circle(pose_6d, oc, radius=5, color=(50, 50, 50), thickness=-1)
            text = "z: {:.0f}".format(int(ozr)) # (int(ozr) if int(ozr) >= 0 else (360+int(ozr)))
            cv2.putText(pose_6d, text, [oc[0]+10, oc[1]-10], cv2.FONT_ITALIC, 0.6, (250, 170, 40), 2)
'''
