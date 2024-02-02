import sys
import threading
import argparse

sys.path.append("src/config")
from modbus_client_configuration import ModbusClientConfiguration
from production_cycle_states_configuration import ProductionCycleStatesConfiguration
from web_app_configuration import WebAppConfiguration
from database_configuration import DatabaseConfiguration
from methods_configuration import MethodsConfiguration
from cobot_offset_position_configuration import CobotOffsetPositionConfiguration
from mqtt_client_configuration import MqttClientConfiguration

sys.path.append("src/services/camera/depthcamera")
from realsense_d435i import RealsenseD435I
sys.path.append("src/methods/neuralnetworks/detection/yolov4")
from yolo_v4 import YoloV4
sys.path.append("src/methods/neuralnetworks/poseestimation/augmentedautoencoder")
from augmented_autoencoder import AugmentedAutoencoder
sys.path.append("src/methods/imageprocessing/pickingpoints")
from object_picking_point import ObjectPickingPoint

sys.path.append("src/services/thread/methods")
from methods_mqtt_thread import MethodsMqttThread


def methods_mqtt():
    real_sense_d4351 = RealsenseD435I((MethodsConfiguration.WIDTH, MethodsConfiguration.HEIGHT), False)
    real_sense_d4351.initialize()
    yolo_v4 = YoloV4(MethodsConfiguration.PATH, False)
    augmented_autoencoder = AugmentedAutoencoder(MethodsConfiguration.PATH, False)
    picking_point = ObjectPickingPoint(False)

    methods_mqtt_thread = MethodsMqttThread(
        real_sense_d4351, yolo_v4, augmented_autoencoder, picking_point,
        MqttClientConfiguration,
        CobotOffsetPositionConfiguration
    )

    methods_mqtt_thread.run()


def main(args):
    try:
        if args.cycle == "methods_mqtt":
            methods_mqtt()
        else:
            print("Source images not correct")
            exit(-1)

    except Exception as e:
        sys.stderr.write(str(e) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cycle", required=False, help="methods_mqtt", default="methods_mqtt")
    args = parser.parse_args()
    main(args)


