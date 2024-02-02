import os
import sys
import cv2

sys.path.append(os.getcwd())
from src.services.camera.depthcamera.realsense_d435i import RealsenseD435I

depth_camera = RealsenseD435I((640, 480), False)
depth_camera.photo("dataset/test/rgb/camera_settings.png")
rgb_image = cv2.imread("dataset/test/rgb/camera_settings.png")
cv2.imshow("rgb_image", rgb_image)
cv2.waitKey()
depth_camera.release()
