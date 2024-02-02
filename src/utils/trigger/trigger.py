class Trigger:
    def __init__(self):
        self.app = False

    def trig(self, value):
        if value and not self.app:
            self.app = True
            return True
        if not value:
            self.app = False
            return False
