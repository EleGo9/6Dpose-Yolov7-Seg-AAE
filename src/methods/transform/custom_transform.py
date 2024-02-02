from torchvision.transforms import *


class CustomTransform:
    def __init__(self, size: tuple):
        self.size = size

        self.compose = None

        self.initialize()

    def initialize(self):
        self.compose = Compose([
            Resize(self.size)
        ])

    def __call__(self, *args, **kwargs):
        return self.compose(*args, **kwargs)

