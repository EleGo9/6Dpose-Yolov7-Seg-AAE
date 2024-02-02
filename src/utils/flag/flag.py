import sys

sys.path.append("src/utils/flag")
from interface_flag import IFlag


class Flag(IFlag):
    def __init__(self):
        self.flag = None

        self.initialize()

    def initialize(self):
        self.flag = False

    def set_flag(self, flag=True):
        self.flag = flag

    def get_flag(self):
        return self.flag

    def reset_flag(self, flag=False):
        self.flag = flag
