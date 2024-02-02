import sys

sys.path.append("src/utils/cycle")
from interface_cycle import ICycle


class ProductionCycle(ICycle):
    def __init__(self, shared_production_variables):
        self.shared_production_variables = shared_production_variables

        self.retained_object_type_index = 0
        self.retained_placed_objects = 0

        self.retained_object_type_end_effector_index = 0
        self.retained_object_type_box_position = []
        self.retained_object_type_number = 0

        self.initialize()

    def initialize(self):
        self.reset()
        self.update()

    def get_object_type_end_effector_index(self):
        return self.retained_object_type_end_effector_index

    def get_object_type_box_position(self):
        return self.retained_object_type_box_position

    def get_object_type_number(self):
        return self.retained_object_type_number

    def reset(self):
        self.shared_production_variables.placed_objects = 0
        self.shared_production_variables.done_bags = 0

    def update(self):
        self.shared_production_variables.current_object = self.shared_production_variables.recipe_objects_types[self.retained_object_type_index]
        self.retained_object_type_end_effector_index = self.shared_production_variables.recipe_end_effector_indexes[self.retained_object_type_index]
        self.retained_object_type_box_position = self.shared_production_variables.recipe_object_types_box_positions[self.retained_object_type_index]
        self.retained_object_type_number = self.shared_production_variables.recipe_object_types_numbers[self.retained_object_type_index]

    def start(self):
        start_stop_cycle = self.shared_production_variables.start_stop_cycle
        if start_stop_cycle:
            self.update()
        return start_stop_cycle

    def progress(self):
        self.retained_placed_objects += 1
        self.shared_production_variables.placed_objects += 1

    def control(self):
        self.progress()
        if self.retained_placed_objects == self.retained_object_type_number:
            self.retained_placed_objects = 0
            self.retained_object_type_index += 1
            if self.retained_object_type_index == len(self.shared_production_variables.recipe_objects_types):
                self.shared_production_variables.done_bags += 1
                self.shared_production_variables.placed_objects = 0
                self.retained_object_type_index = 0
                if self.completed():
                    self.stop()
                    self.reset()
                else:
                    self.update()
            else:
                self.update()

    def completed(self):
        completed_production_cycle = self.shared_production_variables.done_bags == self.shared_production_variables.total_bags
        return completed_production_cycle

    def stop(self):
        self.shared_production_variables.start_stop_cycle = False
