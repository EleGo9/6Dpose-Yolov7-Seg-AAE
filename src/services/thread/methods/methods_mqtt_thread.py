import base64
import sys
import cv2
import numpy as np
import math
from time import time, sleep
from queue import Queue

sys.path.append("src/services/thread")
from base_thread import BaseThread
sys.path.append("src/services/mqtt")
from mqtt_client import MqttClient
sys.path.append("src/utils/conversions/angles")
from angles_conversions import AnglesConversions
sys.path.append("src/utils/image")
from draw_angles_on_image import DrawAnglesOnImage
sys.path.append("src/utils/format/jsonsenml")
from format_json_senml import FormatJsonSenML

sys.path.append("src/resources/models")
from resources_mapper import ResourcesMapper

base_topic = "/iot/user/271968@studenti.unimore.it/test"


class MethodsMqttThread(BaseThread):
    def __init__(
            self,
            camera, detection_method, pose_estimation_method, picking_point_method,
            mqtt_client_configuration,
            cobot_position_offset_configuration
    ):
        super().__init__()
        self.camera = camera
        self.detection_method = detection_method
        self.pose_estimation_method = pose_estimation_method
        self.picking_point_method = picking_point_method
        self.mqtt_client_configuration = mqtt_client_configuration
        self.cobot_position_offset_configuration = cobot_position_offset_configuration

        self.rgb_image = None
        self.depth_image = None
        self.boxes = []
        self.scores = []
        self.labels = []
        self.all_pose_estimates = []
        self.all_class_idcs = []
        self.all_cs = []
        self.all_translation_px = []
        self.singles_renders = []
        self.objects_center = []
        self.euler_angles_z_rotations = []
        self.euler_angles_to_show = []
        self.rotations = []
        self.chosen_object_index = 0
        self.rotation_vector = []
        self.xyz_mm = []

        self.retained_detection_image = None
        self.retained_renders_image = None
        self.retained_6d_pose_estimation_image = None
        self.retained_camK = None

        self.request_6d_pose_estimation_queue = Queue(maxsize=1)

        self.resources_mapper = None
        self.format_json_senml = None
        self.mqtt_client = None

        self.initialize()

        self.publish_info()

    def initialize(self):
        self.resources_mapper = ResourcesMapper(base_topic=self.mqtt_client_configuration.BASE_TOPIC)
        self.format_json_senml = FormatJsonSenML()
        self.mqtt_client = MqttClient(self.mqtt_client_configuration)

        self.mqtt_client.connect()
        self.mqtt_client.on_connect()
        self.subscribe_all_topic()
        self.mqtt_client.start()

    def subscribe_all_topic(self):
        self.mqtt_client.standard.subscribe_control(
            self.resources_mapper.get_resource("request_6d_pose_estimation").get_topic(),
            queue=self.request_6d_pose_estimation_queue
        )

    def run(self):
        while True:
            '''if self.request_6d_pose_estimation_queue.full():
                message = self.request_6d_pose_estimation_queue.get()
                resource = self.resources_mapper.get_resource("request_6d_pose_estimation")
                self.format_json_senml.from_format(message, resource)
                if resource.get_value():'''
            start_time = time()
            self.photo()
            self.detection()
            self.pose_estimation()
            while len(self.all_pose_estimates) == 0:
                self.photo()
                self.detection()
                self.pose_estimation()
            self.picking_point()
            self.pose_refinements()
            self.pose_6d_mm_rv()
            self.show()
            end_time = time()
            inference_time = round(end_time-start_time, 3)
            print("6d pose estimation: {:.3f} s".format(inference_time))
            print()

            resource = self.resources_mapper.get_resource("6d_pose_estimation")
            resource.set_value(self.shared_variables.pose_6d)
            message = self.format_json_senml.to_format(resource)
            self.mqtt_client.standard.publish_control_response(
                resource.get_topic(),
                message,
                resource.get_qos(),
                resource.get_retained()
            )
            print(message.to_json())

            resource = self.resources_mapper.get_resource("confidence")
            resource.set_value([round(v*100, 1) for v in self.scores])
            message = self.format_json_senml.to_format(resource)
            self.mqtt_client.standard.publish_telemetry(
                resource.get_topic(),
                message,
                resource.get_qos(),
                resource.get_retained()
            )
            print(message.to_json())

            resource = self.resources_mapper.get_resource("cosine_similarity")
            resource.set_value([int(v*1000)/10 for v in self.all_cs])
            message = self.format_json_senml.to_format(resource)
            self.mqtt_client.standard.publish_telemetry(
                resource.get_topic(),
                message,
                resource.get_qos(),
                resource.get_retained()
            )
            print(message.to_json())

            resource = self.resources_mapper.get_resource("inference_time")
            resource.set_value(inference_time)
            message = self.format_json_senml.to_format(resource)
            self.mqtt_client.standard.publish_telemetry(
                resource.get_topic(),
                message,
                resource.get_qos(),
                resource.get_retained()
            )
            print(message.to_json())

            #sleep(10)
            continue

            sleep(0.05)

            bytes_image = base64.b64encode(self.rgb_image)
            rgb_image = bytes_image.decode("ascii")
            resource = self.resources_mapper.get_resource("rgb_image")
            resource.set_value(rgb_image)
            message = self.format_json_senml.to_format(resource)
            self.mqtt_client.standard.publish_telemetry(
                resource.get_topic(),
                message,
                resource.get_qos(),
                resource.get_retained()
            )

            bytes_image = base64.b64encode(self.retained_detection_image)
            detection_image = bytes_image.decode("ascii")
            resource = self.resources_mapper.get_resource("detection_image")
            resource.set_value(detection_image)
            message = self.format_json_senml.to_format(resource)
            self.mqtt_client.standard.publish_telemetry(
                resource.get_topic(),
                message,
                resource.get_qos(),
                resource.get_retained()
            )

            bytes_image = base64.b64encode(self.retained_6d_pose_estimation_image)
            pose_6d_image = bytes_image.decode("ascii")
            resource = self.resources_mapper.get_resource("6d_pose_estimation_image")
            resource.set_value(pose_6d_image)
            message = self.format_json_senml.to_format(resource)
            self.mqtt_client.standard.publish_telemetry(
                resource.get_topic(),
                message,
                resource.get_qos(),
                resource.get_retained()
            )

            bytes_image = base64.b64encode(self.retained_renders_image)
            picking_point_image = bytes_image.decode("ascii")
            resource = self.resources_mapper.get_resource("picking_point_image")
            resource.set_value(picking_point_image)
            message = self.format_json_senml.to_format(resource)
            self.mqtt_client.standard.publish_telemetry(
                resource.get_topic(),
                message,
                resource.get_qos(),
                resource.get_retained()
            )

            '''import base64
            #rgb_image_byte_array = base64.b64encode(self.rgb_image)
            # rgb_image = str(rgb_image_byte_array)
            with open("src/services/webapp/static/img/detection.png") as file:
                pass #rgb_image = file.read()

            base64_bytes = base64.b64encode(self.rgb_image)
            base64_bytes = base64_bytes.decode('ascii')

            img = base64_bytes.encode('ascii')
            rgb_image = base64.b64decode(img)

            #rgb_image_byte_array = str.encode(rgb_image)
            #rgb_image = np.array(list(rgb_image_byte_array))
            #rgb_image = np.resize(rgb_image, (480, 640, 3)).astype(np.uint8)
            #rgb_image = base64.b64decode(rgb_image_byte_array)
            rgb_image = np.fromstring(rgb_image, dtype=np.uint8)
            print(rgb_image.shape)

            rgb_image = np.resize(rgb_image, (480, 640, 3))
            print(self.rgb_image.shape, rgb_image.shape)
            full_image = np.concatenate((self.rgb_image, rgb_image), axis=1)
            cv2.imshow("Results", full_image)
            cv2.waitKey()
            cv2.destroyAllWindows()'''

    def photo(self):
        self.camera.photo("src/images/camera.png")
        self.rgb_image = cv2.imread("src/images/camera_backup.png")
        # self.rgb_image = cv2.imread("src/images/camera_backup.png")

    def detection(self):
        self.boxes, self.scores, self.labels = self.detection_method.detect(self.rgb_image)
        self.retained_detection_image = self.detection_method.draw(self.rgb_image)
        print("YOLO scores:", self.scores)
        print("YOLO mean score:", np.round(np.mean(self.scores), 4))

    def pose_estimation(self):
        self.boxes, self.scores, self.labels, self.all_pose_estimates, self.all_class_idcs, self.all_cs, self.all_translation_px = \
            self.pose_estimation_method.pose_estimation(self.rgb_image, self.labels, self.boxes, self.scores)
        print("AAE cs:", self.all_cs)
        print("AAE mean cs:", np.round(np.mean(self.all_cs), 4))

        self.retained_6d_pose_estimation_image, _ = self.pose_estimation_method.draw(
            self.rgb_image,
            self.all_pose_estimates, self.all_class_idcs,
            self.labels, self.boxes, self.scores, self.all_cs
        )
        cv2.imwrite("src/services/webapp/static/img/6d_pose_estimation.png", self.retained_6d_pose_estimation_image)
        # cv2.imshow("6d_pose_estimation", pose_estimation_image)
        # cv2.waitKey()

        self.retained_renders_image, self.singles_renders, self.retained_camK = self.pose_estimation_method.singles_renders(
            self.rgb_image,
            self.all_pose_estimates, self.all_class_idcs,
            self.labels, self.boxes, self.scores, self.all_cs
        )

    def picking_point(self):
        self.objects_center = []
        for sr in self.singles_renders:
            self.objects_center.append(self.picking_point_method.find(sr))

    def pose_refinements(self):
        pass

    def pose_6d_mm_rv(self):
        # self.shared_variables.pose_6d = [6120, -1340, 700, 8000, 31415, 0]
        self.choose_object()
        self.xyz_pixel_to_xyz_mm()
        self.euler_angles_z_rotations_adjustment()
        self.euler_angles_to_rotation_vector()
        self.offset_adjustment()

        self.shared_variables.pose_6d = [
            self.xyz_mm[1], self.xyz_mm[0], self.xyz_mm[2],
            self.rotation_vector[0], self.rotation_vector[1], self.rotation_vector[2]
        ]
        print("xyz [mm]: {}".format(self.xyz_mm))
        print("rxryrz [rv]: {}".format(np.round(np.rad2deg(self.rotation_vector))))
        print("6d pose: {}".format(self.shared_variables.pose_6d))

    def choose_object(self):
        self.chosen_object_index = 0

    def xyz_pixel_to_xyz_mm(self):
        picking_point = self.objects_center[self.chosen_object_index]
        picking_point_depth = self.camera.depth("src/images/depth.png", picking_point)
        x_mm, y_mm, z_mm = self.camera.homography("coordinates_to_mm", picking_point, picking_point_depth)
        x_mm = int(x_mm*10)
        y_mm = int(y_mm*10)
        z_mm = int(z_mm*10)
        self.xyz_mm = [x_mm, y_mm, z_mm]

    def euler_angles_z_rotations_adjustment(self):
        self.euler_angles_z_rotations = []
        self.euler_angles_to_show = []
        self.rotations = []
        for pe in self.all_pose_estimates:
            '''z_rotation = -AnglesConversions.rotation_matrix_to_euler_angles(pe[:3, :3])[2]
            if z_rotation < -np.pi/2:
                z_rotation += np.pi
            elif z_rotation > np.pi/2:
                z_rotation -= np.pi
            self.euler_angles_z_rotations.append(z_rotation)'''

            rotation_vector, _ = cv2.Rodrigues(pe[:3, :3])
            points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
            axis_points, _ = cv2.projectPoints(
                points,
                np.array(rotation_vector.reshape(1, 3)[0]),
                np.array(pe[:3, 3]).T,
                self.retained_camK, (0, 0, 0, 0)
            )

            b = axis_points[3].ravel()
            c = axis_points[1].ravel()
            a = [b[0] + 50, b[1]]
            self.rotations.append([
                tuple(axis_points[0].astype(int).ravel()),
                tuple(axis_points[1].astype(int).ravel()),
                tuple(axis_points[2].astype(int).ravel())
            ])

            z_angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))

            z_angle_to_show = z_angle
            if z_angle_to_show > 0:
                z_angle_to_show = np.pi/2 - z_angle_to_show
            elif z_angle_to_show < 0:
                z_angle_to_show = -(np.pi/2 + z_angle_to_show)
            self.euler_angles_to_show.append(z_angle_to_show)

            z_rotation = np.deg2rad(z_angle)
            if z_rotation > 0:
                z_rotation = np.pi/2 - z_rotation
            elif z_rotation < 0:
                z_rotation = -(np.pi/2 + z_rotation)
            self.euler_angles_z_rotations.append(z_rotation)

    def euler_angles_to_rotation_vector(self):
        euler_angle = [0, np.pi, self.euler_angles_z_rotations[self.chosen_object_index]]
        self.rotation_vector = AnglesConversions.euler_angles_to_rotation_vector(euler_angle)
        self.rotation_vector = [int(10000*i) for i in self.rotation_vector]

    def offset_adjustment(self):
        self.xyz_mm[0] = self.cobot_position_offset_configuration.X_BASE_TO_END_EFFECTOR \
                         + self.cobot_position_offset_configuration.X_END_EFFECTOR_TO_CAMERA \
                         - self.xyz_mm[0]
        self.xyz_mm[1] = self.cobot_position_offset_configuration.Y_BASE_TO_END_EFFECTOR \
                         + self.cobot_position_offset_configuration.Y_END_EFFECTOR_TO_CAMERA \
                         - self.xyz_mm[1]
        self.xyz_mm[2] = self.cobot_position_offset_configuration.Z_BASE_TO_END_EFFECTOR \
                         + self.cobot_position_offset_configuration.Z_END_EFFECTOR_TO_CAMERA \
                         - 3460 #self.xyz_mm[2]
        '''5120 + 1070 - self.xyz_mm[1],  # int(y_mm*10),
        -320 + 200 - self.xyz_mm[0],  # int(x_mm*10),
        2650 + 1400 - 346 * 10,  # self.xyz_mm[2],'''

    def show(self, vis=True):
        DrawAnglesOnImage.draw(self.retained_renders_image, self.objects_center, self.euler_angles_to_show, self.rotations)
        full_image = np.concatenate(
            (self.retained_detection_image, self.retained_6d_pose_estimation_image, self.retained_renders_image),
            axis=1
        )
        cv2.imwrite("src/images/picking_points.png", full_image)
        cv2.imwrite("src/services/webapp/static/img/picking_points.png", self.retained_renders_image)
        if vis:
            cv2.imshow("Results", full_image)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def publish_info(self):
        resource = self.resources_mapper.get_resource("info")
        resource.set_value(["6D pose estimation", "Hakertz Lab", "v1.7"])
        message = self.format_json_senml.to_format(resource)
        self.mqtt_client.standard.publish_info(
            resource.get_topic(),
            message,
            resource.get_qos(),
            resource.get_retained()
        )
