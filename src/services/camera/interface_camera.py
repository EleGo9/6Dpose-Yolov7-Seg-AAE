import abc


class ICamera(abc.ABC):
    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def calibrate(self):
        pass

    @abc.abstractmethod
    def photo(self, rgb_path):
        pass

    @abc.abstractmethod
    def release(self):
        pass
