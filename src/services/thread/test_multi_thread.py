import os
import sys
import threading
import argparse

sys.path.append(os.getcwd())
from src.config.modbus_client_configuration import ModbusClientConfiguration
from src.config.production_cycle_states_configuration import ProductionCycleStatesConfiguration
from src.config.web_app_configuration import WebAppConfiguration
from src.config.database_configuration import DatabaseConfiguration
from src.config.methods_configuration import MethodsConfiguration
from src.config.cobot_offset_position_configuration import CobotOffsetPositionConfiguration
from src.config.mqtt_client_configuration import MqttClientConfiguration

'''
sys.path.append("src/services/webapp")
from web_app import WebApp'''
from src.services.modbus.modbus_client import ModbusClient
from src.services.camera.depthcamera.realsense_d435i import RealsenseD435I
from src.methods.neuralnetworks.segmentation.u_mask_rcnn.u_mask_rcnn import UMaskRCNN
from src.methods.neuralnetworks.segmentation.yolov7.yolo_v7 import Yolov7
from src.methods.neuralnetworks.poseestimation.augmentedautoencoder.augmented_autoencoder import AugmentedAutoencoder

sys.path.append("src/methods/imageprocessing/pickingpoints")
from src.methods.imageprocessing.pickingpoint.object_picking_point import ObjectPickingPoint
'''sys.path.append("src/services/mqtt/publisher")
from mqtt_publisher_client import MqttPublisherClient'''

'''from src.services.thread.webapp.web_app_thread import WebAppThread
from src.services.thread.mqtt.mqtt_thread import MqttThread'''
from src.services.thread.modbus.modbus_thread import ModbusThread
from src.services.thread.methods.methods_thread import MethodsThread
from src.services.thread.methods.only_methods_thread import OnlyMethodsThread
from src.services.thread.webapp.simulate_web_app_thread import SimulateWebAppThread


def pipeline():
    modbus_client = ModbusClient(ModbusClientConfiguration.IP_ADDRESS, ModbusClientConfiguration.PORT)
    modbus_thread = ModbusThread(modbus_client, ModbusClientConfiguration.REGISTERS_ADDRESS, ProductionCycleStatesConfiguration)

    real_sense_d4351 = RealsenseD435I((MethodsConfiguration.WIDTH, MethodsConfiguration.HEIGHT), False)
    real_sense_d4351.initialize()
    yolo_v4 = YoloV4(MethodsConfiguration.PATH, False)
    augmented_autoencoder = AugmentedAutoencoder(MethodsConfiguration.PATH, False)
    picking_point = ObjectPickingPoint(False)
    methods_thread = MethodsThread(real_sense_d4351, yolo_v4, augmented_autoencoder, picking_point, CobotOffsetPositionConfiguration)
    #methods_thread = MethodsThread(None, None, None, None)

    web_app = WebApp(WebAppConfiguration.IP_ADDRESS, WebAppConfiguration.PORT, WebAppConfiguration.DEBUG, DatabaseConfiguration.PATH)
    web_app_thread = WebAppThread(web_app)

    topics = [
        "start_stop_cycle/telemetry",
        "selected_recipe/telemetry",
        "done_bags/telemetry",
        "current_object/telemetry",
        "cobot_state_cycle/telemetry",
        "safety/telemetry"
    ]
    mqtt_publisher_client = MqttPublisherClient(MqttClientConfiguration)
    mqtt_thread = MqttThread(mqtt_publisher_client, MqttClientConfiguration.INTERVAL_S, topics)

    modbus_threading = threading.Thread(target=modbus_thread.run)
    methods_threading = threading.Thread(target=methods_thread.run)
    web_app_threading = threading.Thread(target=web_app_thread.run)
    mqtt_threading = threading.Thread(target=mqtt_thread.run)


    modbus_threading.start()
    #methods_threading.start()
    web_app_threading.start()
    mqtt_threading.start()

    methods_thread.run()

    modbus_threading.join()
    methods_threading.join()
    web_app_threading.join()
    mqtt_threading.join()


def only_methods():
    real_sense_d4351 = RealsenseD435I((MethodsConfiguration.WIDTH, MethodsConfiguration.HEIGHT), False)
    real_sense_d4351.initialize()

    # u_mask_rcnn = UMaskRCNN("src/config/mask_rcnn.yml")
    yolo_v7 = Yolov7(MethodsConfiguration.PATH)
    augmented_autoencoder = AugmentedAutoencoder(MethodsConfiguration.PATH, False)
    picking_point = ObjectPickingPoint(False)

    only_methods_thread = OnlyMethodsThread(real_sense_d4351, yolo_v7, augmented_autoencoder, picking_point, CobotOffsetPositionConfiguration)

    only_methods_thread.run()


def loop_pick_and_place():
    modbus_client = ModbusClient(ModbusClientConfiguration.IP_ADDRESS, ModbusClientConfiguration.PORT)
    modbus_thread = ModbusThread(modbus_client, ModbusClientConfiguration.REGISTERS_ADDRESS, ProductionCycleStatesConfiguration)

    real_sense_d4351 = RealsenseD435I((MethodsConfiguration.WIDTH, MethodsConfiguration.HEIGHT), False)
    real_sense_d4351.initialize()
    # u_mask_rcnn = UMaskRCNN("src/config/mask_rcnn.yml")
    yolo_v7 = Yolov7(MethodsConfiguration.PATH)
    augmented_autoencoder = AugmentedAutoencoder(MethodsConfiguration.PATH, False)
    picking_point = ObjectPickingPoint(False)
    methods_thread = MethodsThread(real_sense_d4351, yolo_v7, augmented_autoencoder, picking_point, CobotOffsetPositionConfiguration)

    simulate_web_app_thread = SimulateWebAppThread()

    modbus_threading = threading.Thread(target=modbus_thread.run)
    methods_threading = threading.Thread(target=methods_thread.run)
    simulate_web_app_thread = threading.Thread(target=simulate_web_app_thread.run)


    modbus_threading.start()
    #methods_threading.start()
    simulate_web_app_thread.start()

    methods_thread.run()

    modbus_threading.join()
    methods_threading.join()
    simulate_web_app_thread.join()


def main(args):
    if args.cycle == "pipeline":
        pipeline()
    elif args.cycle == "pnp":
        loop_pick_and_place()
    elif args.cycle == "methods":
        only_methods()
    else:
        print("Source images not correct")
        exit(-1)
    try:
        pass
    except Exception as e:
        sys.stderr.write(str(e) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cycle", required=False, help="pipeline, pnp, methods", default="methods")
    args = parser.parse_args()
    main(args)


