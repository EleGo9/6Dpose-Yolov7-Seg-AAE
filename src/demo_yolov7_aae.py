import cv2
import glob
import numpy as np
import os
import time
import argparse
import configparser
import csv
from augmentedautoencoder import AugmentedAutoencoder
from yolov7_seg import YoloV7
from PIL import Image
import os
import struct
import numpy as np
import json

import yaml

def xyxy2xywh(xyxy):
    (x1, y1) = (int(xyxy[0]), int(xyxy[1]))
    (x2, y2) = (int(xyxy[2]), int(xyxy[3]))
    w = int(x2-x1)
    h = int(y2-y1)
    return [x1, y1, w, h]

def demo_folder(args, test_configpath):
    n_classes = 5
    file_path = args.file_path
    filedepth_path = args.filedepth_path
    save_res = args.save_res
    seg_yes = args.seg_yes
    if os.path.isdir(file_path):
        files = sorted \
            (glob.glob(os.path.join(str(file_path) ,'*.png') ) +glob.glob(os.path.join(str(file_path) ,'*.jpg')))
    else:
        files = [file_path]

    if os.path.isdir(filedepth_path):
        depthfiles = sorted(glob.glob(os.path.join(str(filedepth_path) ,'*.png') ) +glob.glob
            (os.path.join(str(filedepth_path) ,'*.jpg')))
    elif filedepth_path != '':
        depthfiles = [filedepth_path]
    else:
        depthfiles = None

    if save_res != '':
        if not os.path.exists(save_res):
            os.makedirs(save_res)

    test_args = configparser.ConfigParser()
    test_args.read(test_configpath)
    icp_flag = False
    if (filedepth_path != '' and test_args.has_option('ICP' ,'icp')):
        icp_flag = test_args.getboolean('ICP' ,'icp')

    aae = AugmentedAutoencoder(test_configpath, args.debugvis, icp_flag)
    seg = YoloV7(test_configpath, args.debugvis)


    for file in files:
        image0 = cv2.imread(file)
        (H, W) = image0.shape[:2]
        # print(image.shape)
        # cv2.imshow('im', image)
        # cv2.waitKey(0)
        image = np.swapaxes(image0, 0, 2)
        image = np.swapaxes(image, 1, 2)
        # print(image.shape)
        # cv2.imshow('im',image)
        # cv2.waitKey(0)

        # Apply detection and segmentation
        labels, scores, boxes, masks, im_masks = seg.segment(image, image, False)
        # labels = [label+1 for label in labels]
        print(labels)
        aae_boxes = []
        for box in boxes:
            aae_boxes.append(xyxy2xywh(box))
        # yolo_im = seg.draw(image0)
        # cv2.imshow('yolo image', yolo_im)
        # cv2.waitKey(0)

        # Apply masks
        masks = masks.detach().cpu().numpy()
        unified_mask = masks[0]
        for i in range(1,masks.shape[0]):
            unified_mask = cv2.bitwise_or(unified_mask, masks[i], mask = None)
        unified_mask = unified_mask.astype('uint8')
        masked = cv2.bitwise_and(image0, image0, mask=unified_mask)
        # cv2.imshow('masked image', masked)
        # cv2.waitKey(0)
        yolo_det = seg.draw(image0)
        if seg_yes:
            yolo_det_seg = seg.draw(masked)
        else:
            yolo_det_seg=yolo_det
        # yolo_im = np.concatenate((yolo_det, yolo_det_seg), axis=1)
        # cv2.imshow('yolo image', yolo_im)
        # cv2.waitKey(0)



        # Estimate the 6D pose
        start_time = time.time()

        # For the moment don't take into account chiave fissa because there are problems with the ply
        if 1 in labels:
            ind = labels.index(1)
            labels.pop(ind)
            aae_boxes.pop(ind)
            scores.pop(ind)

        aae_boxes, scores, labels = aae.process_detection_output(H, W, aae_boxes, scores, labels)
        if seg_yes:
            all_pose_estimates, all_class_idcs, _ = aae.process_pose(aae_boxes, labels, masked)
        else:
            all_pose_estimates, all_class_idcs, _ = aae.process_pose(aae_boxes, labels, image0)
        aae_im = aae.draw(image0, all_pose_estimates, all_class_idcs, aae_boxes, scores, labels, [])
        pose_estimation = np.concatenate((image0, yolo_det_seg, aae_im), axis=1 )
        cv2.imshow('6D pose estimation', pose_estimation)
        cv2.waitKey(0)

        #Save images to a folder
        cv2.imwrite(save_res+str(file[-6:]), pose_estimation)

        # To see the image:
        # cv2.imshow('im', im_mask)
        # cv2.waitKey(0)

        # cropped_img = seg.crop_mask(image0, pred, im_mask)


    # undistortion preparatory steps
    # mtx = np.array(eval(test_args.get('CAMERA' ,'K_test'))).reshape(3 ,3)
    # dist = None
    # if (test_args.has_option('CAMERA' ,'C_test')):
    #     dist = np.array(eval(test_args.get('CAMERA' ,'C_test'))).reshape(5 ,1)
    # else:
    #     print("WARNING: no distortion coefficients were provided")
    #
    # if isinstance(dist, np.ndarray) and depthfiles!=None:
    #     print("Error: You can't launch ICP with the undistort.")
    #     exit(-1)
    #
    # if depthfiles == None:
    #     depthfiles = [ None for f in files]
    #
    # if len(files) != len(depthfiles):
    #     print("Error: the RGB and depth images must be the same amount.")
    #     exit(-1)
    #
    # first_iter = True
    #
    # for file, depthfile in zip(files, depthfiles):
    #     print(file)
    #     image = cv2.imread(file)
    #     depth = None
    #     if depthfile:
    #         # depth = np.asarray(Image.open(depthfile)).astype(np.float32)
    #         depth = np.asarray(Image.open(depthfile))
    #     (H, W) = image.shape[:2]
    #     # undistortion
    #     if first_iter and isinstance(dist, np.ndarray):
    #         newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (W ,H), 1, (W ,H))
    #         # for remap opdebugvision
    #         mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (W ,H), 5)
    #         first_iter = False

        # undistort is 5/10 times slower than remap
        # undist_image = cv2.undistort(image, np.array(K_test), dist, None, newcameramtx)
        # end_time = time.time()
        # print("--------- undistort standard: {:.3f} s".format(end_time - start_time))
        # if isinstance(dist, np.ndarray):
        #     undist_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        # else:
        #     undist_image = image
        #
        # boxes, scores, labels = net.detect(undist_image)
        # yolo_im = net.draw(undist_image)
        #
        # boxes, scores, labels = aae.process_detection_output(H, W, boxes, scores, labels)
        # all_pose_estimates, all_class_idcs, cosine_similarity = aae.process_pose(boxes, labels, undist_image, depth)
        # # try:
        # aae_im = aae.draw(undist_image, all_pose_estimates, all_class_idcs, boxes, scores, labels, cosine_similarity)
        # # except:
        # #     aae_im = undist_image.copy()
        # #     print(file)
        #
        # full_image = np.concatenate((image, yolo_im, aae_im), axis=1)
        # if args.vis:
        #     cv2.imshow('Results', full_image)
        #     cv2.waitKey(0)
        # elif save_res != '':
        #     im_name = file.split('/')[-1]
        #     cv2.imwrite(save_res + im_name, full_image)

def main(args):
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    if workspace_path == None:
        print('Please define a workspace path:')
        print('export AE_WORKSPACE_PATH=/path/to/workspace')
        exit(-1)

    test_configpath = os.path.join(workspace_path ,'cfg_eval' ,args.test_config)

    if args.file_path != '':
        demo_folder(args, test_configpath)
    # elif args.input != '':  # if args.input:
    #     demo_bag(args, test_configpath)
    # elif args.realsense:  # if args.input:
    #     demo_realsense(args, test_configpath)
    # elif args.video_path != '':
    #     demo_video(args, test_configpath)
    # else:
    #     demo_webcam(args, test_configpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", required=False, help='folder or filename to image(s)', default='')
    parser.add_argument("--seg_yes", help='with (True) or without (False) segmentation',default=True)
    parser.add_argument("-d", "--filedepth_path", required=False, help='folder or filename to depth image(s)', default='')
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file", default='')
    parser.add_argument("-v", "--video_path", required=False, help='filename to test video', default='')
    parser.add_argument("-r", "--realsense", required=False, help='filename to test video', default='')
    parser.add_argument("-test_config", type=str, required=False, default='test_config_webcam.cfg')
    parser.add_argument("-save_res", type=str, required=False, default='')
    parser.add_argument("-vis", action='store_true', default=False)
    parser.add_argument("-debugvis", action='store_true', default=False)
    args = parser.parse_args()
    main(args)