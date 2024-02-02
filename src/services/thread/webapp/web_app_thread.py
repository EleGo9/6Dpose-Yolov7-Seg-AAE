import sys

sys.path.append("src/services/thread")
from base_thread import BaseThread


class WebAppThread(BaseThread):
    def __init__(self, web_app):
        super().__init__()
        self.web_app = web_app
        self.web_app.set_shared_pages_variables(self.shared_variables)

    def initialize(self):
        pass

    def run(self):
        self.web_app.run()
        '''self.shared_variables.start_stop_cycle = True
        self.shared_variables.recipe_objects_types = [""]
        self.shared_variables.recipe_end_effector_indexes = [0]
        self.shared_variables.recipe_object_types_box_positions = [[5120, -320, 2650, 0, 31415, 0]]
        self.shared_variables.recipe_object_types_numbers = [10]'''



