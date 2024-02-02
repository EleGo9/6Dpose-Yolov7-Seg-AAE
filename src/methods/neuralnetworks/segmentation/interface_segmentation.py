import abc


class ISegmentation(abc.ABC):

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def segment(self, image):
        pass

    @abc.abstractmethod
    def draw(self, image, labels, scores, boxes, masks):
        pass

    @abc.abstractmethod
    def mask(self, image, masks):
        pass
