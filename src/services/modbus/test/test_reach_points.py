from pymodbus.client.sync import ModbusTcpClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.payload import BinaryPayloadBuilder

import time
import math
import threading

import real_sense_camera
import detection_pose_estimation
import keypoints

path = "/home/hakertz-test/Scrivania/AugmentedAutoencoder/custom/"
path_image = path + "image/"
cfg_file = "custom/ugello_l80_99.cfg"
detection_path = "/home/hakertz-test/Scrivania/darknet/custom_train/"
detection_weights = detection_path + "yolov4_ugello_l80_99_final.weights"
detection_cfg_file = detection_path + "yolov4_ugello_l80_99.cfg"
detection_data_file = detection_path + "cifarelli_001.data"
detection_thresh = 0.4
path_camera_calib = path + "calib_camera/"
image_name = path_image + "camera_photo.jpg"
image_show = False
path_save = path_image

client = None
picking_points = None

def communication():
    setup()
    loop()


def setup():
    global client, picking_points
    client = ModbusTcpClient("192.168.1.100", port=502)
    real_sense_camera.initialize()
    real_sense_camera.photo(image_name)
    detector_pose_estimator = detection_pose_estimation.DetectionPoseEstimation(cfg_file,
                                                                                detection_weights,
                                                                                detection_cfg_file,
                                                                                detection_data_file,
                                                                                detection_thresh,
                                                                                path_camera_calib,
                                                                                path_save,
                                                                                image_show
                                                                                )
    labels, indexes, scores, boxes, translations, rotations, bboxes_rot = detector_pose_estimator.detect_and_pose_estimate(image_name)
    grasping_points = keypoints.grasping_keypoints(bboxes_rot, path_image, image_show)
    picking_points = real_sense_camera.pickable_points(grasping_points, path_image, image_show)

    print(translations, rotations)
    print(grasping_points)
    print(picking_points)


def loop():
    global client
    if client.connect():
        connect_robot()
    client.close()


def connect_robot():
    global client, picking_points
    # [120, 30, 160] offset rgb cam - picking center
    
    position = [4260+1200-int(picking_points[0]['punti'][0][1]*10),
                -320+300-int(picking_points[0]['punti'][0][0]*10),
                3220+1600-int(picking_points[0]['punti'][0][2]*10),
                0, int(10000*abs(math.radians(picking_points[0]['asse_rotazione'][2]))), 0]
    print(position)

    encoded_position(129, position)


def encoded_position(mb_base_address, pos):
    for i in range(6):
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)
        builder.add_16bit_int(pos[i])
        payload = builder.to_registers()
        payload = builder.build()
        client.write_registers(mb_base_address+i, payload, unit=1, skip_encode=True)


thread_mb_register = threading.Thread(target=communication)
thread_mb_register.start()
thread_mb_register.join()
