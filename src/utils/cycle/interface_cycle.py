import abc


class ICycle(abc.ABC):

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def get_object_type_end_effector_index(self):
        pass

    @abc.abstractmethod
    def get_object_type_box_position(self):
        pass

    @abc.abstractmethod
    def get_object_type_number(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def control(self):
        pass
