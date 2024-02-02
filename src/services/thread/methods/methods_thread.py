import sys
import cv2
import numpy as np
import math
from time import time

from src.services.thread.base_thread import BaseThread
from src.utils.conversions.angles.angles_conversions import AnglesConversions
from src.utils.image.draw_angles_on_image import DrawAnglesOnImage


class MethodsThread(BaseThread):
    def __init__(self, camera, segmentation_method, pose_estimation_method, picking_point_method, cobot_position_offset_configuration):
        super().__init__()
        self.camera = camera
        self.segmentation_method = segmentation_method
        self.pose_estimation_method = pose_estimation_method
        self.picking_point_method = picking_point_method
        self.cobot_position_offset_configuration = cobot_position_offset_configuration

        self.rgb_image = None
        self.rgb_image_support = None
        self.depth_image = None
        self.boxes = []
        self.scores = []
        self.labels = []
        self.masks = None

        self.all_pose_estimates = []
        self.all_class_idcs = []
        self.all_cosine_similarity = []
        self.all_translation_px = []
        self.singles_renders = []

        self.objects_center = []
        self.euler_angles_z_rotations = []
        self.euler_angles_to_show = []
        self.rotations = []
        self.chosen_object_index = 0
        self.rotation_vector = []
        self.xyz_mm = []

        self.retained_segmentation_image = None
        self.retained_masked_image = None
        self.retained_renders_image = None
        self.retained_6d_pose_estimation_image = None
        self.retained_camK = None
        self.retained_cycle_counter = 0
        self.retained_first_scan = True

        self.initialize()

    def initialize(self):
        pass

    def run(self):
        while True:
            if self.synchronization.start_6d_pose_estimation:
                if not self.synchronization.end_6d_pose_estimation:
                    start_time = time()
                    while len(self.all_pose_estimates) == 0 or self.retained_first_scan:
                        self.retained_first_scan = False
                        self.photo()
                        if self.segmentation() != -1:
                            self.pose_estimation()
                    self.retained_first_scan = True
                    self.picking_point()
                    self.pose_refinements()
                    self.pose_6d_mm_rv()
                    self.show(True)
                    end_time = time()
                    print("6d pose estimation: {:.3f} s".format(end_time - start_time))
                    print()
                    self.synchronization.end_6d_pose_estimation = True
            else:
                self.synchronization.end_6d_pose_estimation = False

    def photo(self):
        self.camera.photo("dataset/test/rgb/camera.png")
        self.rgb_image = cv2.imread("dataset/test/rgb/camera.png")
        # self.rgb_image = cv2.imread("src/images/camera_backup.png")

    def segmentation(self):
        self.labels, self.scores, self.boxes, self.masks = self.segmentation_method.segment(self.rgb_image)

        if len(self.labels) == 0:
            print("No objects detected")
            return -1

        self.retained_masked_image = self.filtering(self.rgb_image)
        self.retained_masked_image = self.segmentation_method.mask(self.retained_masked_image, self.masks)
        self.retained_segmentation_image = self.segmentation_method.draw(self.rgb_image, self.labels, self.scores,
                                                                         self.boxes, self.masks)
        '''# self.rgb_image_support = np.swapaxes(self.rgb_image, 0, 2)
        # self.rgb_image_support = np.swapaxes(self.rgb_image_support, 1, 2)
        self.labels, self.scores, self.boxes, self.masks = self.segmentation_method.segment(
            self.rgb_image)  # self.rgb_image_support, self.rgb_image_support, False)
        if len(self.labels) == 0:
            print("No objects detected", len(self.labels), self.labels)
            return -1
        self.retained_segmentation_image = self.segmentation_method.draw(self.rgb_image, self.labels, self.scores,
                                                                         self.boxes, self.masks)
        # self.retained_masked_image = self.yolo_mask(self.rgb_image, self.masks)
        self.retained_masked_image = self.mask_image(self.rgb_image, self.masks)
        # self.rgb_image = self.rgb_image.transpose(1, 2, 0)
        # self.scores = [s.cpu().numpy() for s in self.scores]
        # print("YOLO scores:", self.scores)
        print("YOLO mean score:", np.round(np.mean(self.scores), 4), np.round(np.std(self.scores), 4))
'''

        '''self.rgb_image_support = np.swapaxes(self.rgb_image, 0, 2)
        self.rgb_image_support = np.swapaxes(self.rgb_image_support, 1, 2)
        self.labels, self.scores, self.boxes, self.masks, self.retained_segmentation_image = self.segmentation_method.segment(self.rgb_image_support, self.rgb_image_support, False)
        if len(self.labels) == 0:
            print("No objects detected", len(self.labels), self.labels)
            return -1
        #self.retained_segmentation_image = self.segmentation_method.draw(self.rgb_image, self.labels, self.scores, self.boxes, self.masks)
        self.retained_masked_image = self.yolo_mask(self.rgb_image, self.masks)
        # self.retained_masked_image = self.mask_image(self.rgb_image, self.masks)
        #self.rgb_image = self.rgb_image.transpose(1, 2, 0)
        self.scores = [s.cpu().numpy() for s in self.scores]
        #print("YOLO scores:", self.scores)
        print("YOLO mean score:", np.round(np.mean(self.scores), 4), np.round(np.std(self.scores), 4))
'''
        '''self.labels, self.scores, self.boxes, self.masks = self.segmentation_method.segment(
            self.rgb_image)  # self.rgb_image_support, self.rgb_image_support, False)
        if len(self.labels) == 0:
            print("No objects detected", len(self.labels), self.labels)
            return -1
        self.retained_segmentation_image = self.segmentation_method.draw(self.rgb_image, self.labels, self.scores,                                                              self.boxes, self.masks)
        # self.retained_masked_image = self.yolo_mask(self.rgb_image, self.masks)
        self.retained_masked_image = self.mask_image(self.rgb_image, self.masks)
        # print("YOLO scores:", self.scores)
        # print("YOLO mean score:", np.round(np.mean(self.scores), 4))'''

    def pose_estimation(self):
        for i in range(len(self.boxes)):
            self.boxes[i] = self.xyxy2xywh(self.boxes[i])

        self.boxes, self.scores, self.labels, self.all_pose_estimates, self.all_class_idcs, self.all_cosine_similarity = \
            self.pose_estimation_method.pose_estimation(self.retained_masked_image, self.labels, self.boxes, self.scores)

        self.retained_6d_pose_estimation_image, _ = self.pose_estimation_method.draw(
            self.rgb_image,
            self.all_pose_estimates, self.all_class_idcs,
            self.labels, self.boxes, self.scores, self.all_cosine_similarity
        )

        self.retained_renders_image, self.singles_renders, self.retained_camK = self.pose_estimation_method.singles_renders(
            self.rgb_image,
            self.all_pose_estimates, self.all_class_idcs,
            self.labels, self.boxes, self.scores, self.all_cosine_similarity
        )
        '''for i in range(len(self.boxes)):
            self.boxes[i] = self.xyxy2xywh(self.boxes[i])
        self.boxes, self.scores, self.labels, self.all_pose_estimates, self.all_class_idcs, self.all_cosine_similarity = \
            self.pose_estimation_method.pose_estimation(self.retained_masked_image, self.labels, self.boxes, self.scores)
        #print("AAE cs:", self.all_cosine_similarity)
        print("AAE mean cs:", np.round(np.mean(self.all_cosine_similarity), 4), np.round(np.std(self.all_cosine_similarity), 4))

        self.retained_6d_pose_estimation_image, _ = self.pose_estimation_method.draw(
            self.rgb_image,
            self.all_pose_estimates, self.all_class_idcs,
            self.labels, self.boxes, self.scores, self.all_cosine_similarity
        )
        # cv2.imwrite("src/services/webapp/static/img/6d_pose_estimation.png", self.retained_6d_pose_estimation_image)
        # cv2.imshow("6d_pose_estimation", pose_estimation_image)
        # cv2.waitKey()

        self.retained_renders_image, self.singles_renders, self.retained_camK = self.pose_estimation_method.singles_renders(
            self.rgb_image,
            self.all_pose_estimates, self.all_class_idcs,
            self.labels, self.boxes, self.scores, self.all_cosine_similarity
        )'''

    @staticmethod
    def filtering(image):
        image = cv2.bitwise_not(image)
        sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, sharpen_filter)
        return image

    @staticmethod
    def mask_image(image, masks):
        image = cv2.bitwise_not(image)
        sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, sharpen_filter)

        masks = masks.transpose(1, 2, 0)
        masks = masks.max(axis=2)
        masks = np.expand_dims(masks, axis=2)
        masks = masks.astype(np.uint8)

        masked_image = cv2.bitwise_and(image, image, mask=masks)
        # cv2.imshow("image", cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        # cv2.waitKey()
        return masked_image

    @staticmethod
    def yolo_mask(image, masks):
        masks = masks.detach().cpu().numpy()
        unified_mask = masks[0]
        for i in range(1, masks.shape[0]):
            unified_mask = cv2.bitwise_or(unified_mask, masks[i], mask=None)
        unified_mask = unified_mask.astype('uint8')
        # image = image.transpose(1, 2, 0)
        masked = cv2.bitwise_and(image, image, mask=unified_mask)
        return masked

    @staticmethod
    def xyxy2xywh(xyxy):
        (x1, y1) = (int(xyxy[0]), int(xyxy[1]))
        (x2, y2) = (int(xyxy[2]), int(xyxy[3]))
        w = int(x2 - x1)
        h = int(y2 - y1)
        return [x1, y1, w, h]

    def picking_point(self):
        self.objects_center = []
        for sr in self.singles_renders:
            self.objects_center.append(self.picking_point_method.find(sr))

    def pose_refinements(self):
        pass

    def pose_6d_mm_rv(self):
        # self.shared_variables.pose_6d = [6120, -1340, 700, 8000, 31415, 0]
        self.choose_object()
        self.xyz_pixel_to_xyz_mm()
        self.euler_angles_z_rotations_adjustment()
        self.euler_angles_to_rotation_vector()
        self.offset_adjustment()

        self.shared_variables.pose_6d = [
            self.xyz_mm[1], self.xyz_mm[0], self.xyz_mm[2],
            self.rotation_vector[0], self.rotation_vector[1], self.rotation_vector[2]
        ]
        # print("xyz [mm]: {}".format(self.xyz_mm))
        # print("rxryrz [rv]: {}".format(np.round(np.rad2deg(self.rotation_vector))))
        # print("6d pose: {}".format(self.shared_variables.pose_6d))

    def choose_object(self):
        self.chosen_object_index = 0

    def xyz_pixel_to_xyz_mm(self):
        picking_point = self.objects_center[self.chosen_object_index]
        picking_point_depth = self.camera.depth("src/images/depth.png", picking_point)
        x_mm, y_mm, z_mm = self.camera.homography("coordinates_to_mm", picking_point, picking_point_depth)
        x_mm = int(x_mm*10)
        y_mm = int(y_mm*10)
        z_mm = int(z_mm*10)
        self.xyz_mm = [x_mm, y_mm, z_mm]

    def euler_angles_z_rotations_adjustment(self):
        self.euler_angles_z_rotations = []
        self.euler_angles_to_show = []
        self.rotations = []
        for pe in self.all_pose_estimates:
            '''z_rotation = -AnglesConversions.rotation_matrix_to_euler_angles(pe[:3, :3])[2]
            if z_rotation < -np.pi/2:
                z_rotation += np.pi
            elif z_rotation > np.pi/2:
                z_rotation -= np.pi
            self.euler_angles_z_rotations.append(z_rotation)'''

            rotation_vector, _ = cv2.Rodrigues(pe[:3, :3])
            points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
            axis_points, _ = cv2.projectPoints(
                points,
                np.array(rotation_vector.reshape(1, 3)[0]),
                np.array(pe[:3, 3]).T,
                self.retained_camK, (0, 0, 0, 0)
            )

            b = axis_points[3].ravel()
            c = axis_points[1].ravel()
            a = [b[0] + 50, b[1]]
            self.rotations.append([
                tuple(axis_points[0].astype(int).ravel()),
                tuple(axis_points[1].astype(int).ravel()),
                tuple(axis_points[2].astype(int).ravel())
            ])

            z_angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))

            z_angle_to_show = z_angle
            if z_angle_to_show > 0:
                z_angle_to_show = np.pi/2 - z_angle_to_show
            elif z_angle_to_show < 0:
                z_angle_to_show = -(np.pi/2 + z_angle_to_show)
            self.euler_angles_to_show.append(z_angle_to_show)

            z_rotation = np.deg2rad(z_angle)
            if z_rotation > 0:
                z_rotation = np.pi/2 - z_rotation
            elif z_rotation < 0:
                z_rotation = -(np.pi/2 + z_rotation)
            self.euler_angles_z_rotations.append(z_rotation)

    def euler_angles_to_rotation_vector(self):
        euler_angle = [0, np.pi, self.euler_angles_z_rotations[self.chosen_object_index]]
        self.rotation_vector = AnglesConversions.euler_angles_to_rotation_vector(euler_angle)
        self.rotation_vector = [int(10000*i) for i in self.rotation_vector]

    def offset_adjustment(self):
        self.xyz_mm[0] = self.cobot_position_offset_configuration.X_BASE_TO_END_EFFECTOR \
                         + self.cobot_position_offset_configuration.X_END_EFFECTOR_TO_CAMERA \
                         - self.xyz_mm[0]
        self.xyz_mm[1] = self.cobot_position_offset_configuration.Y_BASE_TO_END_EFFECTOR \
                         + self.cobot_position_offset_configuration.Y_END_EFFECTOR_TO_CAMERA \
                         - self.xyz_mm[1]
        type_choosen_object = self.labels[self.chosen_object_index]
        offset = 3460
        if type_choosen_object == 1:
            offset = 3050
        elif type_choosen_object == 3:
            offset = 3200
        self.xyz_mm[2] = self.cobot_position_offset_configuration.Z_BASE_TO_END_EFFECTOR \
                         + self.cobot_position_offset_configuration.Z_END_EFFECTOR_TO_CAMERA \
                         - offset #3460 #self.xyz_mm[2]
        '''5120 + 1070 - self.xyz_mm[1],  # int(y_mm*10),
        -320 + 200 - self.xyz_mm[0],  # int(x_mm*10),
        2650 + 1400 - 346 * 10,  # self.xyz_mm[2],'''

    def show(self, vis=True):
        DrawAnglesOnImage.draw(self.retained_renders_image, self.objects_center, self.euler_angles_to_show,
                               self.rotations)
        full_image = np.concatenate(
            (self.retained_segmentation_image, self.retained_6d_pose_estimation_image, self.retained_renders_image),
            axis=1
        )
        # cv2.imwrite("src/images/picking_points.png", full_image)
        # cv2.imwrite("src/services/webapp/static/img/picking_points.png", self.retained_renders_image)
        if vis:
            cv2.imshow("Results", full_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
