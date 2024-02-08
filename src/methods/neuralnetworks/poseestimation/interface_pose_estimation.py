import abc


class IPoseEstimation(abc.ABC):

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def pose_estimation(self, color_img, filtered_labels, filtered_boxes, masks, scores, depth_img=None, camPose=None):
        pass

    @abc.abstractmethod
    def draw(self, image, all_pose_estimates, all_class_idcs, labels, boxes, scores, cosine_similarities):
        pass
