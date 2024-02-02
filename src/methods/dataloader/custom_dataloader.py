from torch.utils.data import DataLoader


class CustomDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(CustomDataloader, self).__init__(*args, **kwargs)

        self.initialize()

    def initialize(self):
        pass
