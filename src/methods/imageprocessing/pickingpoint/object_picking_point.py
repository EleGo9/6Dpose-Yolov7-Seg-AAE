import sys
import numpy as np
import cv2
import imutils
import time


class ObjectPickingPoint:
    def __init__(self, debug_vis=False):
        self.debug_vis = debug_vis

    def initialize(self):
        pass

    def find(self, image, index=None):
        start_time = time.time()
        pre_processed_image = self.pre_processing(image)
        center = self.contours_center(pre_processed_image, index)
        end_time = time.time()
        # print("Object Picking Point \"Detection\" Time: {:.3f} s".format(end_time - start_time))
        return center

    def pre_processing(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        kernel = np.ones((6, 6), np.uint8)
        dilate = cv2.dilate(opening, kernel, iterations=3)
        blur = cv2.blur(dilate, (15, 15))
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def contours_center(self, image, index):
        contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) == 0:
            return np.array((0, 0)) # image.shape[:2]
        M = cv2.moments(contours[0])
        if M["m00"] == 0:
            return np.array((0, 0)) # image.shape[:2]

        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        contours_image = cv2.drawContours(image, contours, 0, (30, 200, 255), 2)
        if self.debug_vis:
            if index is not None:
                cv2.imshow("Object Contours " + str(index), contours_image)
            cv2.imshow("Object contours", contours_image)

        return [center_x, center_y]
