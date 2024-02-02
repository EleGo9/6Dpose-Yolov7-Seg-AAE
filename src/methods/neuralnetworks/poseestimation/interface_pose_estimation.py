import abc


class IPoseEstimation(abc.ABC):

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def pose_estimation(self, filtered_boxes, filtered_labels, color_img, depth_img=None, camPose=None):
        pass

    @abc.abstractmethod
    def draw(self, image):
        pass
