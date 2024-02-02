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
sys.path.append("src/services/mqtt")
from mqtt_client import MqttClient

sys.path.append("src/services/thread/methods")
from methods_thread import MethodsThread
sys.path.append("src/services/thread/mqtt")
from mqtt_methods_thread import MqttMethodsThread


def mqtt_methods():
    real_sense_d4351 = RealsenseD435I((MethodsConfiguration.WIDTH, MethodsConfiguration.HEIGHT), False)
    real_sense_d4351.initialize()
    yolo_v4 = YoloV4(MethodsConfiguration.PATH, False)
    augmented_autoencoder = AugmentedAutoencoder(MethodsConfiguration.PATH, False)
    picking_point = ObjectPickingPoint(False)
    methods_thread = MethodsThread(real_sense_d4351, yolo_v4, augmented_autoencoder, picking_point, CobotOffsetPositionConfiguration)
    #methods_thread = MethodsThread(None, None, None, None)

    mqtt_client = MqttClient(MqttClientConfiguration)
    mqtt_methods_thread = MqttMethodsThread(mqtt_client)

    methods_threading = threading.Thread(target=methods_thread.run)
    mqtt_methods_threading = threading.Thread(target=mqtt_methods_thread.run)


    #methods_threading.start()
    mqtt_methods_threading.start()

    methods_thread.run()

    methods_threading.join()
    mqtt_methods_threading.join()


def main(args):
    try:
        if args.cycle == "mqtt_methods":
            mqtt_methods()
        else:
            print("Source images not correct")
            exit(-1)

    except Exception as e:
        sys.stderr.write(str(e) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cycle", required=False, help="mqtt_methods", default="mqtt_methods")
    args = parser.parse_args()
    main(args)


