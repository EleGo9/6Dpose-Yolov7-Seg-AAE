import abc


class IFlag(abc.ABC):

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def set_flag(self, flag=True):
        pass

    @abc.abstractmethod
    def get_flag(self):
        pass

    @abc.abstractmethod
    def reset_flag(self, flag=False):
        pass

