import os
import sys
sys.path.append(os.getcwd())

from src.methods.model.sg_det import SGDet
from src.services.methods.metrics.metrics_segmentation import MetricsSegmentation


model = SGDet(num_channels=3, num_classes=4)

metrics = MetricsSegmentation("src/config/metrics_segmentation.yml")
metrics.define_transform()
metrics.define_dataset()
metrics.define_dataloader()
metrics.define_model(model)
metrics.compute()

print("Metrics finished")
