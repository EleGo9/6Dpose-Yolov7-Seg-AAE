import os
import sys
sys.path.append(os.getcwd())

from src.methods.model.faster_rcnn import FasterRCNNModel
from src.services.methods.metrics.metrics_detection import MetricsDetection

model = FasterRCNNModel(num_classes=4)
model = model.return_model()

metrics = MetricsDetection("src/config/metrics_detection.yml")
metrics.define_transform()
metrics.define_dataset()
metrics.define_dataloader()
metrics.define_model(model)
metrics.compute()

print("Metrics finished")
