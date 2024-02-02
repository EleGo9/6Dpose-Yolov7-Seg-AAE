import time
import cv2
import numpy as np

from src.services.thread.base_thread import BaseThread


class MethodsPipelineThread(BaseThread):
    def __init__(self, camera, segmentation_method, pose_estimation_method):
        super().__init__()
        self.camera = camera
        self.segmentation_method = segmentation_method
        self.pose_estimation_method = pose_estimation_method

        self.rgb_image = None
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

        self.retained_segmentation_image = None
        self.retained_masked_image = None
        self.retained_renders_image = None
        self.retained_6d_pose_estimation_image = None
        self.retained_cycle_counter = 0
        self.retained_first_scan = True

        self.initialize()

    def initialize(self):
        pass

    def run(self):
        while True:
            start_time = time.time()
            while len(self.all_pose_estimates) == 0 or self.retained_first_scan:
                self.retained_first_scan = False
                self.photo()
                if self.segmentation() != -1:
                    self.pose_estimation()
            self.retained_first_scan = True
            self.show()
            end_time = time.time()
            print("6d pose estimation: {:.3f} s".format(end_time - start_time))

    def photo(self):
        self.camera.photo("dataset/test/rgb/camera.png")
        self.rgb_image = cv2.imread("dataset/test/rgb/camera.png")
        '''self.rgb_image = cv2.imread(
            "dataset/test/objects/mix/{:04d}.png".format(self.retained_cycle_counter)
        )'''
        self.retained_cycle_counter += 1

    def segmentation(self):
        self.labels, self.scores, self.boxes, self.masks = self.segmentation_method.segment(self.rgb_image)

        if len(self.labels) == 0:
            print("No objects detected")
            return -1

        self.retained_masked_image = self.filtering(self.rgb_image)
        self.retained_masked_image = self.segmentation_method.mask(self.retained_masked_image, self.masks)
        self.retained_segmentation_image = self.segmentation_method.draw(self.rgb_image, self.labels, self.scores, self.boxes, self.masks)

        # print("YOLO mean scores:", self.scores)
        # print("YOLO std score:", np.round(np.mean(self.scores), 4), np.round(np.std(self.scores), 4))

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

        # print("AAE mean cosine similarity:", self.all_cosine_similarity)
        # print("AAE std cosine similarity:", np.round(np.mean(self.all_cosine_similarity), 4), np.round(np.std(self.all_cosine_similarity), 4))

    @staticmethod
    def filtering(image):
        image = cv2.bitwise_not(image)
        sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, sharpen_filter)
        return image

    @staticmethod
    def xyxy2xywh(xyxy):
        (x1, y1) = (int(xyxy[0]), int(xyxy[1]))
        (x2, y2) = (int(xyxy[2]), int(xyxy[3]))
        w = int(x2 - x1)
        h = int(y2 - y1)
        return [x1, y1, w, h]

    def show(self, vis=True):
        full_image = np.concatenate(
            (self.retained_segmentation_image, self.retained_6d_pose_estimation_image, self.retained_renders_image),
            axis=1
        )
        cv2.imwrite("src/predictions/results.png", full_image)
        if vis:
            cv2.imshow("Results", full_image)
            cv2.waitKey(1)
