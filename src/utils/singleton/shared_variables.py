import sys

sys.path.append("src/utils/singleton")
from singleton import Singleton


class SharedVariables(metaclass=Singleton):
    def __init__(self):
        self.start_stop_cycle = False
        self.done_bags = 0
        self.total_bags = 0
        self.placed_objects = 0
        self.current_object = ""

        self.selected_recipe = ""
        self.recipe_objects_types = [""]
        self.recipe_end_effector_indexes = [0]
        self.recipe_object_types_box_positions = [[5120, -320, 2650, 0, 31370, 0]]
        self.recipe_object_types_numbers = [0]

        self.cobot_state_cycle = ""
        self.robot_mode = ""
        self.power_on_robot = False
        self.safety = ""
        self.joint_angles = [0, 0, 0, 0, 0, 0]
        self.joint_angle_velocities = [0, 0, 0, 0, 0, 0]
        self.joints_current_consumption_ma = [0, 0, 0, 0, 0, 0]
        self.joints_temperature_c = [0, 0, 0, 0, 0, 0]
        self.tcp = [0, 0, 0, 0, 0, 0]

        self.pose_6d = [0, 0, 0, 0, 0, 0]
