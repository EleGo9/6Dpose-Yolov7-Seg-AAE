import sys

sys.path.append("src/services/thread")
from interface_thread import IThread
sys.path.append("src/utils/singleton")
from synchronization import Synchronization
from shared_variables import SharedVariables


class BaseThread(IThread):
    def __init__(self):
        self.synchronization = Synchronization()
        self.shared_variables = SharedVariables()

    def initialize(self):
        pass

    def run(self):
        pass
