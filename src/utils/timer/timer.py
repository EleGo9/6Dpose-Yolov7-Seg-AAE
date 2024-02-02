from time import time


class Timer:
    def __init__(self, pause_time_ms):
        self.pause_time_ms = pause_time_ms
        self.start_time_ms = 0

    def set(self):
        self.start_time_ms = time()*1000

    def expired(self, pause_time_ms=None):
        if pause_time_ms is not None:
            self.pause_time_ms = pause_time_ms
        if self.start_time_ms + self.pause_time_ms <= time()*1000:
            self.set()
            return True
        return False
