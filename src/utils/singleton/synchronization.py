import sys

sys.path.append("src/utils/singleton")
from singleton import Singleton


class Synchronization(metaclass=Singleton):
    def __init__(self):
        self.start_6d_pose_estimation = False
        self.end_6d_pose_estimation = False
