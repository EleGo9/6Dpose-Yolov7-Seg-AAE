# import cv2
# import glob
# import numpy as np
# import os
# import time
# import argparse
# import cv2
# import glob
# import numpy as np
# import os
# import time
# import argparse
# import configparser
# import csv
# from augmentedautoencoder import AugmentedAutoencoder
# from yolov7_seg import YoloV7
# from PIL import Image
# import os
# import struct
# import numpy as np
# import json
#
# import yaml
#
# def xyxy2xywh(xyxy):
#     (x1, y1) = (int(xyxy[0]), int(xyxy[1]))
#     (x2, y2) = (int(xyxy[2]), int(xyxy[3]))
#     w = int(x2-x1)
#     h = int(y2-y1)
#     return [x1, y1, w, h]
#
# def eval_folder(args, test_configpath, out_file):
#     n_classes = 5
#     file_path = args.file_path
#     filedepth_path = args.filedepth_path
#     save_res = args.save_res
#     seg_yes = args.seg_yes
#     if os.path.isdir(file_path):
#         files = sorted \
#             (glob.glob(os.path.join(str(file_path), '*.png')) + glob.glob(os.path.join(str(file_path), '*.jpg')))
#     else:
#         files = [file_path]
#
#     if os.path.isdir(filedepth_path):
#         depthfiles = sorted(glob.glob(os.path.join(str(filedepth_path), '*.png')) + glob.glob
#         (os.path.join(str(filedepth_path), '*.jpg')))
#     elif filedepth_path != '':
#         depthfiles = [filedepth_path]
#     else:
#         depthfiles = None
#
#     if save_res != '':
#         if not os.path.exists(save_res):
#             os.makedirs(save_res)
#
#     test_args = configparser.ConfigParser()
#     test_args.read(test_configpath)
#     icp_flag = False
#     if (filedepth_path != '' and test_args.has_option('ICP', 'icp')):
#         icp_flag = test_args.getboolean('ICP', 'icp')
#
#     aae = AugmentedAutoencoder(test_configpath, args.debugvis, icp_flag)
#     seg = YoloV7(test_configpath, args.debugvis)
#     results = {}
#     results['scene_id'] = []
#     results['im_id'] = []
#     results['obj_id'] = []
#     results['score'] = []
#     results['R'] = []
#     results['t'] = []
#     results['time'] = []
#
#     for file in files:
#         image0 = cv2.imread(file)
#         (H, W) = image0.shape[:2]
#         # print(image.shape)
#         # cv2.imshow('im', image)
#         # cv2.waitKey(0)
#         image = np.swapaxes(image0, 0, 2)
#         image = np.swapaxes(image, 1, 2)
#         # print(image.shape)
#         # cv2.imshow('im',image)
#         # cv2.waitKey(0)
#
#         # Apply detection and segmentation
#         labels, scores, boxes, masks, im_masks = seg.segment(image, image, False)
#         # labels = [label+1 for label in labels]
#         print(labels)
#         aae_boxes = []
#         for box in boxes:
#             aae_boxes.append(xyxy2xywh(box))
#         # yolo_im = seg.draw(image0)
#         # cv2.imshow('yolo image', yolo_im)
#         # cv2.waitKey(0)
#
#         # Apply masks
#         masks = masks.detach().cpu().numpy()
#         unified_mask = masks[0]
#         for i in range(1, masks.shape[0]):
#             unified_mask = cv2.bitwise_or(unified_mask, masks[i], mask=None)
#         unified_mask = unified_mask.astype('uint8')
#         masked = cv2.bitwise_and(image0, image0, mask=unified_mask)
#         # cv2.imshow('masked image', masked)
#         # cv2.waitKey(0)
#         yolo_det = seg.draw(image0)
#         if seg_yes:
#             yolo_det_seg = seg.draw(masked)
#         else:
#             yolo_det_seg = yolo_det
#         if args.debugvis:
#             yolo_im = np.concatenate((yolo_det, yolo_det_seg), axis=1)
#             cv2.imshow('yolo image', yolo_im)
#             cv2.waitKey(0)
#
#         # Estimate the 6D pose
#         start_time = time.time()
#
#         # For the moment don't take into account chiave fissa because there are problems with the ply
#         # if 1 in labels:
#         #     ind = labels.index(1)
#         #     labels.pop(ind)
#         #     aae_boxes.pop(ind)
#         #     scores.pop(ind)
#         if 0 in labels:
#             ind = labels.index(0)
#             labels= [labels[ind]]
#             aae_boxes = [aae_boxes[ind]]
#             scores = [scores[ind]]
#         else:
#             labels = []
#             aae_boxes = []
#             scores = []
#
#
#         aae_boxes, scores, labels = aae.process_detection_output(H, W, aae_boxes, scores, labels)
#         if seg_yes:
#             all_pose_estimates, all_class_idcs, _ = aae.process_pose(aae_boxes, labels, masked)
#         else:
#             all_pose_estimates, all_class_idcs, _ = aae.process_pose(aae_boxes, labels, image0)
#         aae_im = aae.draw(image0, all_pose_estimates, all_class_idcs, aae_boxes, scores, labels, [])
#         # Uncomment if you want to see results
#         # im = np.concatenate((masked, aae_im), axis=1)
#         # cv2.imshow('Results', im)
#         # cv2.waitKey(0)
#         # pose_estimation = np.concatenate((image0, yolo_det_seg, aae_im), axis=1)
#         end_time = time.time()
#
#         for label, box, score, ae_pose, ae_label in zip(labels, boxes, scores, all_pose_estimates, all_class_idcs):
#             if labels:
#                 results['scene_id'].append(label)
#                 results['im_id'].append(int(file.split('/')[-1].split('.')[0].split('_')[-1]))
#                 assert label == ae_label
#                 results['obj_id'].append(label)
#                 results['score'].append(score.detach().cpu().numpy())
#                 results['R'].append(ae_pose[:3, :3])
#                 results['t'].append(ae_pose[:3, 3])
#                 results['time'].append(end_time - start_time)
#
#     # save prova custom test csv
#     f = open(out_file, 'w')
#     writer = csv.writer(f)
#     writer.writerow(results.keys())
#     for i in range(len(results['scene_id'])):
#         R = np.array(results['R'][i]).flatten()
#         t = np.array(results['t'][i]).flatten()
#         R_str = ''
#         for ri in R:
#             R_str = R_str + str(ri) + ' '
#         t_str = ''
#         for ti in t:
#             t_str = t_str + str(ti) + ' '
#         writer.writerow([results['scene_id'][i], results['im_id'][i], results['obj_id'][i], results['score'][i],
#                          R_str, t_str, results['time'][i]])
#
#     f.close()
#
#
# def main(args):
#     workspace_path = os.environ.get('AE_WORKSPACE_PATH')
#     if workspace_path == None:
#         print('Please define a workspace path:')
#         print('export AE_WORKSPACE_PATH=/path/to/workspace')
#         exit(-1)
#
#     test_configpath = os.path.join(workspace_path ,'cfg_eval' ,args.test_config)
#
#     if args.file_path != '':
#         eval_folder(args, test_configpath, args.out_csvfile)
#     # elif args.input != '':  # if args.input:
#     #     demo_bag(args, test_configpath)
#     # elif args.realsense:  # if args.input:
#     #     demo_realsense(args, test_configpath)
#     # elif args.video_path != '':
#     #     demo_video(args, test_configpath)
#     # else:
#     #     demo_webcam(args, test_configpath)
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-f", "--file_path", required=False, help='folder or filename to image(s)', default='')
#     parser.add_argument("--seg_yes", help='with (True) or without (False) segmentation',default=True)
#     parser.add_argument("-d", "--filedepth_path", required=False, help='folder or filename to depth image(s)', default='')
#     parser.add_argument("-i", "--input", type=str, help="Path to the bag file", default='')
#     parser.add_argument("-v", "--video_path", required=False, help='filename to test video', default='')
#     parser.add_argument("-r", "--realsense", required=False, help='filename to test video', default='')
#     parser.add_argument("-test_config", type=str, required=False, default='test_config_webcam.cfg')
#     parser.add_argument("-save_res", type=str, required=False, default='')
#     parser.add_argument("-vis", action='store_true', default=False)
#     parser.add_argument("-debugvis", action='store_true', default=False)
#     parser.add_argument("-out_csvfile", help='path to the file where results will be saved'  , default='')
#     args = parser.parse_args()
#     main(args)

import cv2
import glob
import numpy as np
import os
import time
import argparse
import cv2
import glob
import numpy as np
import os
import time
import argparse
import configparser
import csv
from augmentedautoencoder import AugmentedAutoencoder
from yolov7_seg import YoloV7seg
from yolov7_det import YoloV7det
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

def eval_folder(args, test_configpath, out_file):
    n_classes = 5
    file_path = args.file_path
    filedepth_path = args.filedepth_path
    save_res = args.save_res
    seg_yes = args.seg_yes
    if os.path.isdir(file_path):
        files = sorted \
            (glob.glob(os.path.join(str(file_path), '*.png')) + glob.glob(os.path.join(str(file_path), '*.jpg')))
    else:
        files = [file_path]

    if os.path.isdir(filedepth_path):
        depthfiles = sorted(glob.glob(os.path.join(str(filedepth_path), '*.png')) + glob.glob
        (os.path.join(str(filedepth_path), '*.jpg')))
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
    if (filedepth_path != '' and test_args.has_option('ICP', 'icp')):
        icp_flag = test_args.getboolean('ICP', 'icp')
    print('Loading augmented autoencoder ...')
    aae = AugmentedAutoencoder(test_configpath, args.debugvis, icp_flag)
    print('Done')
    print('Loading detection weights ...')
    if seg_yes:
        seg = YoloV7seg(test_configpath, args.debugvis)
    else:
        det = YoloV7det(test_configpath, args.debugvis)
    print('Done')

    results = {}
    results['scene_id'] = []
    results['im_id'] = []
    results['obj_id'] = []
    results['score'] = []
    results['R'] = []
    results['t'] = []
    results['time'] = []

    gt = True
    if gt:
        with open("/home/elena/repos/Cifarelli/test/dado/testdepth/data/01/gt.yml", "r") as gt:
            ground_truth_labels = yaml.safe_load(gt)

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
        if seg_yes:
            labels, scores, boxes, masks, im_masks = seg.segment(image, image, False)
        else:
            labels, scores, boxes = det.detect(image, image)

        # labels = [label+1 for label in labels]
        print(labels)
        aae_boxes = []
        # if seg_yes:
        if gt:
            im_id = int(file.split('/')[-1].split('.')[0].split('_')[-1])
            box = ground_truth_labels[im_id][0]['obj_bb']
            aae_boxes.append(box)
        else:
            for box in boxes:
                aae_boxes.append(xyxy2xywh(box))
        # yolo_im = seg.draw(image0)
        # cv2.imshow('yolo image', yolo_im)
        # cv2.waitKey(0)

        # Apply masks
        if seg_yes:
            masks = masks.detach().cpu().numpy()
            unified_mask = masks[0]
            for i in range(1, masks.shape[0]):
                unified_mask = cv2.bitwise_or(unified_mask, masks[i], mask=None)
            unified_mask = unified_mask.astype('uint8')
            masked = cv2.bitwise_and(image0, image0, mask=unified_mask)
        # cv2.imshow('masked image', masked)
        # cv2.waitKey(0)
        if seg_yes:
            yolo_det = seg.draw(image0)
        else:
            yolo_det = det.draw(image0)
        if seg_yes:
            yolo_det_seg = seg.draw(masked)
        else:
            yolo_det_seg = yolo_det
        if args.debugvis:
            yolo_im = np.concatenate((yolo_det, yolo_det_seg), axis=1)
            cv2.imshow('yolo image', yolo_im)
            cv2.waitKey(0)



        # Estimate the 6D pose
        start_time = time.time()

        # For the moment don't take into account chiave fissa because there are problems with the ply
        # if 1 in labels:
        #     ind = labels.index(1)
        #     labels.pop(ind)
        #     aae_boxes.pop(ind)
        #     scores.pop(ind)
        if 2 in labels:
            ind = labels.index(2)
            labels= [labels[ind]]
            labels = [2 for i in range(len(labels))]
            aae_boxes = [aae_boxes[ind]]
            scores = [scores[ind]]
        else:
            labels = []
            aae_boxes = []
            scores = []


        aae_boxes, scores, labels = aae.process_detection_output(H, W, aae_boxes, scores, labels)
        if seg_yes:
            all_pose_estimates, all_class_idcs, _ = aae.process_pose(aae_boxes, labels, masked)
        else:
            all_pose_estimates, all_class_idcs, _ = aae.process_pose(aae_boxes, labels, image0)

        # aae_im = aae.draw(image0, all_pose_estimates, all_class_idcs, aae_boxes, scores, labels, [])
        # # Uncomment if you want to see results
        # pose_estimation = np.concatenate((image0, aae_im), axis=1)
        # cv2.imshow('pose', pose_estimation)
        # cv2.waitKey(0)
        end_time = time.time()

        for label, box, score, ae_pose, ae_label in zip(labels, boxes, scores, all_pose_estimates, all_class_idcs):
            if labels:
                results['scene_id'].append(label)
                results['im_id'].append(int(file.split('/')[-1].split('.')[0].split('_')[-1]))
                assert label == ae_label
                results['obj_id'].append(label)
                results['score'].append(score.detach().cpu().numpy())
                results['R'].append(ae_pose[:3, :3])
                results['t'].append(ae_pose[:3, 3])
                results['time'].append(end_time - start_time)

    # save prova custom test csv
    f = open(out_file, 'w')
    writer = csv.writer(f)
    writer.writerow(results.keys())
    for i in range(len(results['scene_id'])):
        R = np.array(results['R'][i]).flatten()
        t = np.array(results['t'][i]).flatten()
        R_str = ''
        for ri in R:
            R_str = R_str + str(ri) + ' '
        t_str = ''
        for ti in t:
            t_str = t_str + str(ti) + ' '
        writer.writerow([results['scene_id'][i], results['im_id'][i], results['obj_id'][i], results['score'][i],
                         R_str, t_str, results['time'][i]])

    f.close()


def main(args):
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    if workspace_path == None:
        print('Please define a workspace path:')
        print('export AE_WORKSPACE_PATH=/path/to/workspace')
        exit(-1)

    test_configpath = os.path.join(workspace_path ,'cfg_eval' ,args.test_config)

    if args.file_path != '':
        eval_folder(args, test_configpath, args.out_csvfile)
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
    parser.add_argument("--seg_yes", help='with (True) or without (False) segmentation', default=False)
    parser.add_argument("-d", "--filedepth_path", required=False, help='folder or filename to depth image(s)', default='')
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file", default='')
    parser.add_argument("-v", "--video_path", required=False, help='filename to test video', default='')
    parser.add_argument("-r", "--realsense", required=False, help='filename to test video', default='')
    parser.add_argument("-test_config", type=str, required=False, default='test_config_webcam.cfg')
    parser.add_argument("-save_res", type=str, required=False, default='')
    parser.add_argument("-vis", action='store_true', default=False)
    parser.add_argument("-debugvis", action='store_true', default=True)
    parser.add_argument("-out_csvfile", help='path to the file where results will be saved'  , default='')
    args = parser.parse_args()
    main(args)