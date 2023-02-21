import cv2
import numpy as np
import glob
import os
import sys
import time
import configparser
from auto_pose.ae import factory, utils

try:
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
except:
    import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
from tensorflow.compat.v1.keras.backend import set_session
from auto_pose.meshrenderer import meshrenderer, meshrenderer_phong
from auto_pose.ae.utils import get_dataset_path


class AugmentedAutoencoder:
    """ """

    def __init__(self, test_configpath, debug_vis, icp_flag=False):

        self.debug_vis = debug_vis
        test_args = configparser.ConfigParser()
        test_args.read(test_configpath)

        workspace_path = os.environ.get('AE_WORKSPACE_PATH')

        if workspace_path == None:
            print('Please define a workspace path:\n')
            print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
            exit(-1)

        self._camPose = test_args.getboolean('CAMERA', 'camPose')
        self._camK = np.array(eval(test_args.get('CAMERA', 'K_test'))).reshape(3, 3)
        self._width = test_args.getint('CAMERA', 'width')
        self._height = test_args.getint('CAMERA', 'height')

        self._upright = test_args.getboolean('AAE', 'upright')
        self.all_experiments = eval(test_args.get('AAE', 'experiments'))

        self.class_names = eval(test_args.get('DETECTOR', 'class_names'))
        # self.det_threshold = eval(test_args.get('DETECTOR','det_threshold'))
        self.icp = icp_flag

        if self.icp:
            self._depth_scale = test_args.getfloat('DATA', 'depth_scale')

        self.all_codebooks = []
        self.all_train_args = []
        self.pad_factors = []
        self.patch_sizes = []

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = test_args.getfloat('MODEL', 'gpu_memory_fraction')

        self.sess = tf.Session(config=config)
        set_session(self.sess)
        # self.detector = YoloV4(str(test_args.get('DETECTOR','detector_model_path')),
        #                         str(test_args.get('DETECTOR','detector_config_path')), self.det_threshold)

        for i, experiment in enumerate(self.all_experiments):
            full_name = experiment.split('/')
            experiment_name = full_name.pop()
            experiment_group = full_name.pop() if len(full_name) > 0 else ''
            log_dir = utils.get_log_dir(workspace_path, experiment_name, experiment_group)
            ckpt_dir = utils.get_checkpoint_dir(log_dir)
            train_cfg_file_path = utils.get_train_config_exp_file_path(log_dir, experiment_name)
            # train_cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
            train_args = configparser.ConfigParser()
            train_args.read(train_cfg_file_path)
            self.all_train_args.append(train_args)
            self.pad_factors.append(train_args.getfloat('Dataset', 'PAD_FACTOR'))
            self.patch_sizes.append((train_args.getint('Dataset', 'W'), train_args.getint('Dataset', 'H')))

            self.all_codebooks.append(
                factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=False))
            saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=experiment_name))
            factory.restore_checkpoint(self.sess, saver, ckpt_dir)

            # if self.icp:
            #     assert len(self.all_experiments) == 1, 'icp currently only works for one object'
            #     # currently works only for one object
            #     from auto_pose.icp import icp
            #     self.icp_handle = icp.ICP(train_args)
        cad_reconst = [str(train_args.get('Dataset', 'MODEL')) for train_args in self.all_train_args]
        if self.icp:
            from auto_pose.icp import icp
            self.icp_handle = icp.ICP(test_args, self.all_train_args, workspace_path, cad_reconst[0] == 'reconst')

        ply_model_paths = [str(workspace_path) + str(train_args.get('Paths', 'MODEL_PATH'))[1:] for train_args in
                           self.all_train_args]
        cad_reconst[0] = 'reconst'
        if cad_reconst[0] == 'cad':
            self.renderer = meshrenderer.Renderer(
                ply_model_paths,
                samples=1,
                vertex_tmp_store_folder=get_dataset_path(workspace_path),
                vertex_scale=float(1)
            )

        elif cad_reconst[0] == 'reconst':
            self.renderer = meshrenderer_phong.Renderer(
                ply_model_paths,
                samples=1,
                vertex_tmp_store_folder=get_dataset_path(workspace_path),
                vertex_scale=float(1)
            )
        else:
            print("BAD value in MODEL: ", cad_reconst[0])
            exit(-1)
        self.color_dict = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)] * 10

    def extract_square_patch(self, scene_img, bb_xywh, pad_factor, resize=(128, 128), interpolation=cv2.INTER_NEAREST,
                             black_borders=False):

        x, y, w, h = np.array(bb_xywh).astype(np.int32)
        size = int(np.maximum(h, w) * pad_factor)

        left = np.maximum(x + w // 2 - size // 2, 0)
        right = x + w // 2 + size // 2
        top = np.maximum(y + h // 2 - size // 2, 0)
        bottom = y + h // 2 + size // 2
        scene_crop = scene_img[top:bottom, left:right].copy()

        if black_borders:
            scene_crop[:(y - top), :] = 0
            scene_crop[(y + h - top):, :] = 0
            scene_crop[:, :(x - left)] = 0
            scene_crop[:, (x + w - left):] = 0

        scene_crop = cv2.resize(scene_crop, resize, interpolation=interpolation)
        if self.debug_vis:
            cv2.imshow('scene_crop', scene_crop)
            cv2.waitKey(0)
        return scene_crop

    def process_detection_output(self, h, w, boxes, scores, labels):
        start_time = time.time()
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for box, score, label in zip(boxes, scores, labels):
            box[0] = np.minimum(np.maximum(box[0], 0), w)
            box[1] = np.minimum(np.maximum(box[1], 0), h)
            box[2] = np.minimum(np.maximum(box[0] + box[2], 0), w)
            box[3] = np.minimum(np.maximum(box[1] + box[3], 0), h)
            bb_xywh = np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]])
            if bb_xywh[2] < 0 or bb_xywh[3] < 0:
                continue

            filtered_boxes.append(bb_xywh)
            filtered_scores.append(score)
            filtered_labels.append(label)
        end_time = time.time()
        print("AAE Preprocessing Time: {:.3f} s".format(end_time - start_time))
        return (filtered_boxes, filtered_scores, filtered_labels)

    def nearest_rotation(self, clas_idx, session, x, top_n=1, upright=False, return_idcs=False):
        # R_model2cam

        if x.dtype == 'uint8':
            x = x / 255.
        if x.ndim == 3:
            x = np.expand_dims(x, 0)

        cosine_similarity = session.run(self.all_codebooks[clas_idx].cos_similarity,
                                        {self.all_codebooks[clas_idx]._encoder.x: x})
        if top_n == 1:
            if upright:
                idcs = np.argmax(cosine_similarity[:, ::int(self.all_codebooks[clas_idx]._dataset._kw['num_cyclo'])],
                                 axis=1) * int(self.all_codebooks[clas_idx]._dataset._kw['num_cyclo'])
            else:
                idcs = np.argmax(cosine_similarity, axis=1)
        else:
            unsorted_max_idcs = np.argpartition(-cosine_similarity.squeeze(), top_n)[:top_n]
            idcs = unsorted_max_idcs[np.argsort(-cosine_similarity.squeeze()[unsorted_max_idcs])]

        cosine_similarity_to_return = [cosine_similarity[0][idx] for idx in idcs]

        if return_idcs:
            return idcs, cosine_similarity_to_return
        else:
            return self.all_codebooks[clas_idx]._dataset.viewsphere_for_embedding[
                       idcs].squeeze(), cosine_similarity_to_return

    def process_pose(self, filtered_boxes, filtered_labels, color_img, depth_img=None, camPose=None):
        start_time = time.time()
        H, W = color_img.shape[:2]

        all_pose_estimates = []
        all_class_idcs = []
        all_codine_similarity = []

        for j, (box_xywh, label) in enumerate(zip(filtered_boxes, filtered_labels)):
            H_est = np.eye(4)

            try:
                clas_idx = self.class_names.index(label)
            except:
                print('%s not contained in config class_names %s', (label, self.class_names))
                continue

            det_img = self.extract_square_patch(color_img,
                                                box_xywh,
                                                self.pad_factors[clas_idx],
                                                resize=self.patch_sizes[clas_idx],
                                                interpolation=cv2.INTER_LINEAR,
                                                black_borders=True)
            _, cosine_similarity = self.nearest_rotation(clas_idx, self.sess, det_img, top_n=1, upright=self._upright,
                                                         return_idcs=True)
            Rs_est, ts_est = self.all_codebooks[clas_idx].auto_pose6d(self.sess,
                                                                      det_img,
                                                                      box_xywh,
                                                                      self._camK,
                                                                      1,
                                                                      self.all_train_args[clas_idx],
                                                                      upright=self._upright)

            R_est = Rs_est.squeeze()
            t_est = ts_est.squeeze()

            if self.icp:
                print("ICP-refinement")
                assert H == depth_img.shape[0]

                depth_crop = self.extract_square_patch(depth_img,
                                                       box_xywh,
                                                       self.pad_factors[clas_idx],
                                                       resize=self.patch_sizes[clas_idx],
                                                       interpolation=cv2.INTER_NEAREST) * self._depth_scale
                R_est_auto = R_est.copy()
                t_est_auto = t_est.copy()

                R_est, t_est = self.icp_handle.icp_refinement(depth_crop, R_est, t_est, self._camK, (W, H),
                                                              clas_idx=clas_idx, depth_only=True)
                _, ts_est = self.all_codebooks[clas_idx].auto_pose6d(self.sess,
                                                                     det_img,
                                                                     box_xywh,
                                                                     self._camK,
                                                                     1,
                                                                     self.all_train_args[clas_idx],
                                                                     upright=self._upright,
                                                                     depth_pred=t_est[2])
                t_est = ts_est.squeeze()
                R_est, _ = self.icp_handle.icp_refinement(depth_crop, R_est, ts_est.squeeze(), self._camK, (W, H),
                                                          clas_idx=clas_idx, no_depth=True)
                # depth_crop = self.extract_square_patch(depth_img,
                #                                     box_xywh,
                #                                     self.pad_factors[clas_idx],
                #                                     resize=self.patch_sizes[clas_idx],
                #                                     interpolation=cv2.INTER_NEAREST)
                # R_est, t_est = self.icp_handle.icp_refinement(depth_crop, R_est, t_est, self._camK, (W,H))

                H_est[:3, 3] = t_est / self._depth_scale  # mm / m
            else:
                H_est[:3, 3] = t_est

            H_est[:3, :3] = R_est
            # print('translation from camera: ',  H_est[:3,3])

            if self._camPose:
                H_est = np.dot(camPose, H_est)

            all_pose_estimates.append(H_est)
            all_class_idcs.append(clas_idx)
            all_codine_similarity.append(cosine_similarity[0])

        end_time = time.time()
        print("AAE Process-pose Time: {:.3f} s".format(end_time - start_time))

        return (all_pose_estimates, all_class_idcs, all_codine_similarity)

    def draw(self, image, all_pose_estimates, all_class_idcs, boxes, scores, labels, cosine_similarity):
        start_time = time.time()
        im = image.copy()
        bgr, depth, _ = self.renderer.render_many(obj_ids=[clas_idx for clas_idx in all_class_idcs],
                                                  W=self._width,
                                                  H=self._height,
                                                  K=self._camK,
                                                  # R = transform.random_rotation_matrix()[:3,:3],
                                                  Rs=[pose_est[:3, :3] for pose_est in all_pose_estimates],
                                                  ts=[pose_est[:3, 3] for pose_est in all_pose_estimates],
                                                  near=10,
                                                  far=10000,
                                                  random_light=False,
                                                  phong={'ambient': 0.4, 'diffuse': 0.8, 'specular': 0.3})

        bgr = cv2.resize(bgr, (self._width, self._height))

        g_y = np.zeros_like(bgr)
        g_y[:, :, 1] = bgr[:, :, 1]
        im_bg = cv2.bitwise_and(im, im, mask=(g_y[:, :, 1] == 0).astype(np.uint8))
        im_show = cv2.addWeighted(im_bg, 1, g_y, 1, 0)

        # cv2.imshow('pred view rendered', pred_view)
        for label, box, score, cs in zip(labels, boxes, scores, cosine_similarity):
            box = box.astype(np.int32)
            xmin, ymin, xmax, ymax = box[0], box[1], box[0] + box[2], box[1] + box[3]
            text = "{}: {:.4f} - {:.4f}".format(label, score, cs)
            cv2.putText(
                im_show, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, self.color_dict[int(label)], 2)
            cv2.rectangle(im_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        # cv2.imshow('', bgr)
        if self.debug_vis:
            cv2.imshow('real', im_show)
            cv2.waitKey(0)
        end_time = time.time()
        print("AAE Drawing Time: {:.3f} s".format(end_time - start_time))
        return im_show

