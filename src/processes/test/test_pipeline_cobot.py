import os
import sys
import threading
import argparse

sys.path.append(os.getcwd())
from src.config.modbus_client_configuration import ModbusClientConfiguration
from src.config.production_cycle_states_configuration import ProductionCycleStatesConfiguration
from src.config.methods_configuration import MethodsConfiguration
from src.config.cobot_offset_position_configuration import CobotOffsetPositionConfiguration

from src.services.modbus.modbus_client import ModbusClient
from src.services.camera.depthcamera.realsense_d435i import RealsenseD435I
from src.services.camera.depthcamera import realsense_d435i_old
from src.methods.neuralnetworks.segmentation.yolov7.yolo_v7 import Yolov7
from src.methods.neuralnetworks.poseestimation.augmentedautoencoder.augmented_autoencoder import AugmentedAutoencoder
from src.methods.imageprocessing.pickingpoint.object_picking_point import ObjectPickingPoint

from src.services.thread.modbus.modbus_thread import ModbusThread
from src.services.thread.methods.pick_and_place_thread import PickAndPlaceThread
from src.services.thread.webapp.simulate_web_app_thread import SimulateWebAppThread
from src.services.thread.methods.only_methods_thread import OnlyMethodsThread


def only_methods():
    real_sense_d4351 = RealsenseD435I(MethodsConfiguration.PATH, True)

    yolo_v7 = Yolov7(MethodsConfiguration.PATH)
    augmented_autoencoder = AugmentedAutoencoder(MethodsConfiguration.PATH, False)
    picking_point = ObjectPickingPoint(False)

    only_methods_thread = OnlyMethodsThread(real_sense_d4351, yolo_v7, augmented_autoencoder, picking_point, CobotOffsetPositionConfiguration)

    only_methods_thread.run()


def loop_pick_and_place():
    modbus_client = ModbusClient(ModbusClientConfiguration.IP_ADDRESS, ModbusClientConfiguration.PORT)
    modbus_thread = ModbusThread(modbus_client, ModbusClientConfiguration.REGISTERS_ADDRESS, ProductionCycleStatesConfiguration)

    real_sense_d4351 = RealsenseD435I(MethodsConfiguration.PATH, True)
    yolo_v7 = Yolov7(MethodsConfiguration.PATH)
    augmented_autoencoder = AugmentedAutoencoder(MethodsConfiguration.PATH, False)
    picking_point = ObjectPickingPoint(False)
    methods_thread = PickAndPlaceThread(real_sense_d4351, yolo_v7, augmented_autoencoder, picking_point, CobotOffsetPositionConfiguration)

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


def main():
    if args.cycle == "pnp":
        loop_pick_and_place()
    elif args.cycle == "methods":
        only_methods()
    try:
        pass
    except Exception as e:
        sys.stderr.write(str(e) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cycle", required=False, help="pipeline, pnp, methods", default="pnp")
    args = parser.parse_args()
    main()


