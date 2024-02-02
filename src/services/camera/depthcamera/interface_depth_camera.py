import abc
import os
import sys

#BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(BASE_PATH)

from src.services.camera.interface_camera import *


class IDepthCamera(ICamera):
    @abc.abstractmethod
    def depth(self, depth_path, coordinates):
        pass

    @abc.abstractmethod
    def homography(self, direction, value, depth):
        pass
