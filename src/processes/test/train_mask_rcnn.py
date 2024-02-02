import os
import sys
import argparse

sys.path.append(os.getcwd())
from src.methods.model.mask_rcnn import MaskRCNNModel
from src.services.methods.train.train_mask_rcnn import TrainMaskRCNN


def train():
    model = MaskRCNNModel(num_classes=4, mask_predictor=MaskRCNNModel.STANDARD_MASK_RCNN)

    training = TrainMaskRCNN(args.config_file)  # "src/config/train_mask_rcnn.yml"
    training.define_transform()
    training.define_dataset()
    training.define_dataloader()
    training.define_model(model.model, pre_trained=True)
    training.define_optimizer()
    training.train()

    print("Train finished")


def main():
    try:
        train()

    except Exception as e:
        print(e)
        exit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config_file", required=True, help="Path to config file")
    args = parser.parse_args()
    main()
