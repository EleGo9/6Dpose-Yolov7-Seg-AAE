import sys

sys.path.append("src/services/thread")
from base_thread import BaseThread


class SimulateWebAppThread(BaseThread):
    def __init__(self):
        super().__init__()

    def initialize(self):
        pass

    def run(self):
        self.shared_variables.start_stop_cycle = True
        self.shared_variables.recipe_objects_types = [""]
        self.shared_variables.recipe_end_effector_indexes = [0]
        self.shared_variables.recipe_object_types_box_positions = [[5120, -320, 5000, 1650, -31370, 0]] # [[5120, -320, 2650, 1650, -31370, 0]] # [[5120, -320, 950, 0, 31415, 0]]
        self.shared_variables.recipe_object_types_numbers = [1]



