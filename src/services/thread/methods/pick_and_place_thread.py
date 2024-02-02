import sys
import cv2
import numpy as np
import math
from time import time

import torch

from src.services.thread.base_thread import BaseThread
from src.utils.transformation.geometrical_transformation import GeometricalTransformation
from src.utils.conversions.angles.angles_conversions import AnglesConversions
from src.utils.image.draw_angles_on_image import DrawAnglesOnImage
from src.utils.results.results import Results


class PickAndPlaceThread(BaseThread):
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
        self.copied_all_pose_estimates = []
        self.all_class_idcs = []
        self.all_cosine_similarity = []
        self.all_translation_px = []
        self.singles_renders = []

        self.objects_center = []
        self.euler_angles_x_rotations = []
        self.euler_angles_y_rotations = []
        self.euler_angles_z_rotations = []
        self.euler_angles_to_show = []
        self.rotations = []
        self.chosen_object_index = 0
        self.rotation_vector = []
        self.xyz_mm = []
        self.xyz_mm_origin = []
        self.rxryrz_rad = []

        self.retained_segmentation_image = None
        self.retained_masked_image = None
        self.retained_renders_image = None
        self.retained_6d_pose_estimation_image = None
        self.retained_camK = None
        self.retained_cycle_counter = 0
        self.retained_first_scan = True
        self.retained_begin_cycle = True

        self.results = None
        self.save_results = False
        self.cem_cif = True

        self.initialize()

    def initialize(self):
        self.results = Results("src/results/141223/real_aae/vite/", self.save_results)

    def run(self):
        while True:
            if self.synchronization.start_6d_pose_estimation:
                if not self.synchronization.end_6d_pose_estimation:
                    if self.save_results and not self.retained_first_scan:
                        correct_picking = input("correct picking: ")
                        self.results.save_correct_picking(correct_picking)
                    start_time = time()
                    self.results.set_timestamp()
                    while len(self.all_pose_estimates) == 0 or self.retained_begin_cycle:
                        self.retained_begin_cycle = False
                        self.photo()
                        if self.segmentation() != -1:
                            self.processing_segmentation()
                            self.pose_estimation()
                    self.retained_begin_cycle = True
                    self.picking_point_geometric()
                    #self.pose_refinements()
                    if self.pose_6d_mm_rv() == -1:
                        continue
                    self.show(True)
                    end_time = time()
                    # print("6d pose estimation: {:.3f} s".format(end_time - start_time))
                    print()
                    self.retained_first_scan = False
                    self.synchronization.end_6d_pose_estimation = True
            else:
                self.synchronization.end_6d_pose_estimation = False

    def photo(self):
        self.rgb_image = self.camera.photo("src/image/test/camera.png")
        cv2.imwrite("src/image/test/camera.png", self.rgb_image)

        self.results.save_image(self.rgb_image)

    def segmentation(self):
        self.labels, self.scores, self.boxes, self.masks = self.segmentation_method.segment(self.rgb_image)

        # self.remove_indexes()
        if len(self.labels) == 0:
            self.all_pose_estimates = []
            print("No objects detected")
            return -1

        self.retained_segmentation_image = self.segmentation_method.draw(self.rgb_image, self.labels, self.scores, self.boxes, self.masks)
        # self.dilate_mask()
        # self.erode_mask()
        if self.cem_cif:
            self.retained_masked_image = self.segmentation_method.mask(self.rgb_image, self.masks)
        else:
            self.retained_masked_image = self.filtering(self.rgb_image)
            self.retained_masked_image = self.segmentation_method.mask(self.retained_masked_image, self.masks)

        self.results.save_detection(self.labels, self.boxes)
        self.results.save_segmentation(self.masks, self.retained_segmentation_image)
        return 0

    def dilate_mask(self):
        self.masks = self.masks.cpu().numpy().transpose(1, 2, 0)
        kernel = np.ones((5, 5), np.uint8)
        self.masks = cv2.dilate(self.masks, kernel, iterations=1)
        if len(self.masks.shape) == 2:
            self.masks = torch.from_numpy(self.masks).unsqueeze(0)
        else:
            self.masks = torch.from_numpy(self.masks.transpose(2, 0, 1))

    def erode_mask(self):
        self.masks = self.masks.cpu().numpy().transpose(1, 2, 0)
        kernel = np.ones((5, 5), np.uint8)
        self.masks = cv2.erode(self.masks, kernel, iterations=1)
        if len(self.masks.shape) == 2:
            self.masks = torch.from_numpy(self.masks).unsqueeze(0)
        else:
            self.masks = torch.from_numpy(self.masks.transpose(2, 0, 1))

    @staticmethod
    def filtering(image):
        image = cv2.bitwise_not(image)
        sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, sharpen_filter)
        return image

    def remove_indexes(self):
        index_remove = []
        index_keep = []
        for i, l in enumerate(self.labels):
            if l == 3 or l == 1 or l == 0 or l == 2:
                index_remove.append(i)
            else:
                index_keep.append(i)

        for ir in sorted(index_remove, reverse=True):
            self.labels.pop(ir)
            self.scores.pop(ir)
            self.boxes.pop(ir)

        if self.masks is not None:
            self.masks = self.masks[index_keep]

    def processing_segmentation(self):
        pass

    def pose_estimation(self):
        for i in range(len(self.boxes)):
            self.boxes[i] = self.xyxy2xywh(self.boxes[i])

        self.boxes, self.scores, self.labels, self.all_pose_estimates, self.all_class_idcs, self.all_cosine_similarity = \
            self.pose_estimation_method.pose_estimation(self.retained_masked_image, self.labels, self.boxes, self.scores)
        self.copied_all_pose_estimates = self.all_pose_estimates.copy()

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

        self.xyz_mm = self.all_pose_estimates[self.chosen_object_index][:3, 3]
        self.xyz_mm = [int(c*10) for c in self.xyz_mm]
        print("aae", self.xyz_mm)

        self.results.save_6d_pose_estimation(self.all_pose_estimates, self.retained_6d_pose_estimation_image)

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

    def picking_point_geometric(self):
        geometrical_transformation = GeometricalTransformation()
        geometrical_transformation.K = self.retained_camK
        geometrical_transformation.picking_points = [{
            "rotation": [0, 0, 0],
            "translation": [0, 0, 0]
        }]
        for i, (l, pe) in enumerate(zip(self.labels, self.all_pose_estimates)):
            if self.cem_cif:
                if l == 0:
                    geometrical_transformation.picking_points = [{
                        "rotation": [0, 90, 0],
                        "translation": [0, 5, -15]
                    }]
                elif l == 1:
                    geometrical_transformation.picking_points = [{
                        "rotation": [0, 0, 0],
                        "translation": [0, 0, 0]
                    }]
                elif l == 2:
                    geometrical_transformation.picking_points = [{
                        "rotation": [0, 90, 0],
                        "translation": [0, 0, 0]
                    }]
                elif l == 3:
                    geometrical_transformation.picking_points = [{
                        "rotation": [0, 90, 0],
                        "translation": [0, 0, 0]
                    }]
                elif l == 4:
                    geometrical_transformation.picking_points = [{
                        "rotation": [0, 0, 90],
                        "translation": [0, 0, 0]
                    }]
            else:
                if l == 0:
                    geometrical_transformation.picking_points = [{
                        "rotation": [0, -120, 0],
                        "translation": [-10, 5, -5]
                    }]
                    '''geometrical_transformation.picking_points = [{
                        "rotation": [90, 0, 0],
                        "translation": [0, 0, -25]
                    }]'''
                elif l == 1:
                    geometrical_transformation.picking_points = [{
                        "rotation": [0, 0, 0],
                        "translation": [0, -40, 0]
                    }]
                elif l == 2:
                    geometrical_transformation.picking_points = [{
                        "rotation": [0, -120, 0],
                        "translation": [-10, 5, -5]
                    }]

            self.all_pose_estimates[i] = geometrical_transformation.compute(self.all_pose_estimates[i])[0]

        self.objects_center = []
        for pe in self.all_pose_estimates:
            center = geometrical_transformation.center(pe)
            self.objects_center.append(center)

    def pose_refinements(self):
        geometrical_transformation = GeometricalTransformation()
        geometrical_transformation.K = self.retained_camK
        geometrical_transformation.picking_points = [{
            "rotation": [0, 0, 0],
            "translation": [0, 0, 0]
        }]
        for i, (l, pe) in enumerate(zip(self.labels, self.all_pose_estimates)):
            if self.cem_cif:
                if l == 0:
                    geometrical_transformation.picking_points = [{
                        "rotation": [0, 0, 0], #[0, 180, 0]
                        "translation": [0, 0, 0]
                    }]

            self.all_pose_estimates[i] = geometrical_transformation.compute(self.all_pose_estimates[i])[0]

        self.objects_center = []
        for pe in self.all_pose_estimates:
            center = geometrical_transformation.center(pe)
            self.objects_center.append(center)

    def pose_6d_mm_rv(self):
        if self.choose_object() == -1:
            print("Picking point out of image")
            return -1
        self.measure_xyz_pixel_to_xyz_mm()
        # self.estimate_xyz_pixel_to_xyz_mm()
        self.euler_angles_z_rotations_adjustment()
        self.euler_angles_to_rotation_vector()
        self.offset_adjustment()

        # self.rotation_matrix_to_euler_angles()

        self.shared_variables.pose_6d = [
            self.xyz_mm[1], self.xyz_mm[0], self.xyz_mm[2],
            self.rotation_vector[0], self.rotation_vector[1], self.rotation_vector[2]
        ]
        print(self.shared_variables.pose_6d)

        return 0

    def choose_object(self):
        for i, oc in enumerate(self.objects_center):
            if oc[0] < 640 and oc[1] < 480:
                self.chosen_object_index = i
                return 0
        return -1

    def measure_xyz_pixel_to_xyz_mm(self):
        picking_point = self.objects_center[self.chosen_object_index]
        picking_point_depth = self.camera.depth("src/image/test/depth.png", picking_point)
        x_mm, y_mm, z_mm = self.camera.homography("coordinates_to_mm", picking_point, picking_point_depth)
        x_mm = int(x_mm*10)
        y_mm = int(y_mm*10)
        z_mm = int(z_mm*10)
        self.xyz_mm = [x_mm, y_mm, z_mm]
        self.xyz_mm_origin = self.xyz_mm.copy()
        print("real-sense", self.xyz_mm)

    def estimate_xyz_pixel_to_xyz_mm(self):
        self.xyz_mm = self.all_pose_estimates[self.chosen_object_index][:3, 3]
        self.xyz_mm = [int(c*10) for c in self.xyz_mm]
        print("transformation", self.xyz_mm)

    def euler_angles_z_rotations_adjustment(self):
        self.euler_angles_z_rotations = []
        self.euler_angles_to_show = []
        self.rotations = []
        for pe in self.all_pose_estimates:
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

            #x_angle = math.degrees(math.atan2(c[2] - b[2], c[0] - b[0]) - math.atan2(a[2] - b[2], a[0] - b[0]))
            #y_angle = math.degrees(math.atan2(c[2] - b[2], c[1] - b[1]) - math.atan2(a[2] - b[2], a[1] - b[1]))
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
            #self.euler_angles_x_rotations.append(x_angle)
            #self.euler_angles_y_rotations.append(y_angle)
            self.euler_angles_z_rotations.append(z_rotation)

    def euler_angles_to_rotation_vector(self):
        euler_angle = [0, np.pi, self.euler_angles_z_rotations[self.chosen_object_index]]
        '''euler_angle = [
            self.euler_angles_x_rotations[self.chosen_object_index],
            self.euler_angles_y_rotations[self.chosen_object_index],
            self.euler_angles_z_rotations[self.chosen_object_index]
        ]'''
        self.rotation_vector = AnglesConversions.euler_angles_to_rotation_vector(euler_angle)
        self.rotation_vector = [int(10000*i) for i in self.rotation_vector]

    def offset_adjustment(self):
        self.xyz_mm[0] = self.cobot_position_offset_configuration.X_BASE_TO_END_EFFECTOR \
                         + self.cobot_position_offset_configuration.X_END_EFFECTOR_TO_CAMERA \
                         - self.xyz_mm[0]
        self.xyz_mm[1] = self.cobot_position_offset_configuration.Y_BASE_TO_END_EFFECTOR \
                         + self.cobot_position_offset_configuration.Y_END_EFFECTOR_TO_CAMERA \
                         - self.xyz_mm[1]
        self.xyz_mm[2] = self.cobot_position_offset_configuration.Z_BASE_TO_END_EFFECTOR \
                         + self.cobot_position_offset_configuration.Z_END_EFFECTOR_TO_CAMERA \
                         - self.xyz_mm[2]

        if self.labels[self.chosen_object_index] == 0:
            self.xyz_mm[2] -= 130
        elif self.labels[self.chosen_object_index] == 1:
            self.xyz_mm[2] -= 130
        if self.labels[self.chosen_object_index] == 2:
            self.xyz_mm[2] -= 20
        if self.labels[self.chosen_object_index] == 3:
            self.xyz_mm[2] -= 100
        if self.labels[self.chosen_object_index] == 4:
            self.xyz_mm[2] -= 20

            if not self.cem_cif:
                if self.xyz_mm[2] < 700:
                    self.xyz_mm[2] = 700

        if self.xyz_mm[2] < 550:
            print("z", self.xyz_mm[2])
            self.xyz_mm[2] = 550

        self.xyz_mm[2] = 1200

        print("6d pose", self.xyz_mm)

    def rotation_matrix_to_euler_angles(self):
        rotation_matrix = self.all_pose_estimates[self.chosen_object_index][:3, :3]
        print(rotation_matrix)
        rotation_vector = AnglesConversions.rotation_matrix_to_rotation_vector(rotation_matrix)
        self.rotation_vector = [int(10000 * i) for i in rotation_vector]
        print(self.xyz_mm, self.rotation_vector)

    def show(self, vis=True):
        DrawAnglesOnImage.draw(
            self.retained_renders_image,
            self.objects_center, self.euler_angles_to_show, self.rotations,
            self.chosen_object_index
        )

        obj_center = self.camera.homography("mm_to_coordinates", self.xyz_mm_origin)
        print(self.objects_center)
        print(obj_center)
        obj_center = [int(i) for i in obj_center]
        cv2.circle(self.retained_renders_image, obj_center, radius=2, color=(0, 0, 255), thickness=-1)

        full_image_top = np.concatenate((self.retained_segmentation_image, self.retained_masked_image), axis=1)
        full_image_bottom = np.concatenate((self.retained_6d_pose_estimation_image, self.retained_renders_image), axis=1)
        full_image = np.concatenate((full_image_top, full_image_bottom), axis=0)

        self.results.save_picking_point(self.all_pose_estimates, self.xyz_mm, self.retained_renders_image)

        #cv2.imwrite("src/image/test/closer/dado_m5/random/{:03d}.png".format(self.retained_cycle_counter), full_image)
        self.retained_cycle_counter += 1
        if vis:
            cv2.imshow("Results", full_image)
            cv2.waitKey(1000)
            # cv2.destroyAllWindows()

    def renders_comparison(self):
        renders_image = None
        for i in range(len(self.all_pose_estimates)):
            picking_point = self.objects_center[i]
            picking_point_depth = self.camera.depth("src/images/depth.png", picking_point)
            x_mm, y_mm, z_mm = self.camera.homography("coordinates_to_mm", picking_point, picking_point_depth)
            self.copied_all_pose_estimates[i][0][3] = x_mm
            self.copied_all_pose_estimates[i][1][3] = y_mm
            self.copied_all_pose_estimates[i][2][3] = 367-55 #z_mm

            '''renders_image, _ = self.pose_estimation_method.draw(
                self.rgb_image,
                self.copied_all_pose_estimates, self.all_class_idcs,
                self.labels, self.boxes, self.scores, self.all_cosine_similarity
            )'''

            renders_image, _, __ = self.pose_estimation_method.singles_renders(
                self.rgb_image,
                self.copied_all_pose_estimates, self.all_class_idcs,
                self.labels, self.boxes, self.scores, self.all_cosine_similarity,
                alpha=1, beta=0.6
            )

        print("############################################")
        for pe, cpe in zip(self.all_pose_estimates, self.copied_all_pose_estimates):
            print(pe)
            print(cpe)
            print()

        renders_comparison = np.concatenate((self.retained_renders_image, renders_image), axis=1)
        cv2.imshow("renders_comparison", renders_comparison)
        cv2.imwrite("src/image/test/renders_comparison/rc.png", renders_comparison)



