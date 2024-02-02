import sys
from time import time, sleep

sys.path.append("src/services/thread")
from base_thread import BaseThread
sys.path.append("src/resources/models")
from base_mqtt_data_model import BaseMqttDataModel
sys.path.append("src/utils/senml/basic")
from basic_senml_record import BasicSenMLRecord
from basic_senml_pack import BasicSenMLPack


sys.path.append("src/config")
from mqtt_client_configuration import MqttClientConfiguration
sys.path.append("src/services/mqtt")
from mqtt_client import MqttClient

sys.path.append("src/utils/flag")
from interface_flag import IFlag
sys.path.append("src/utils/flag")
from flag import Flag


base_topic = "/iot/user/271968@studenti.unimore.it/fum/hz_lab_mn/plcs/plc_ts17"


class MqttMethodsThread(BaseThread):
    def __init__(self, mqtt_client):
        super().__init__()
        self.mqtt_client = mqtt_client

        self.pose_6d_resource = None
        self.resources = []

        self.flag = Flag()

        self.initialize()


    def initialize(self):
        self.add_resources()
        self.mqtt_client.connect()
        self.mqtt_client.on_connect()
        self.subscribe()

    def run(self):
        self.mqtt_client.loop()

    def add_resources(self):
        self.pose_6d_resource = BaseMqttDataModel(
            "pose_6d",
            base_topic + "/states/cycle/object_present/telemetry",
            0,
            False
        )
        self.resources.append(self.pose_6d_resource)

    def subscribe(self):
        self.mqtt_client.subscribe(
            base_topic + "/states/+/object_present/control",
            self.run_6d_pose_estimation,
            self.flag
        )

    def run_6d_pose_estimation(self, message, flag: IFlag = None):
        print(flag.get_flag())
        print(flag.set_flag())
        message_payload = str(message.payload.decode("utf-8"))
        print(message.topic, message_payload)

        if flag.get_flag():
            self.synchronization.start_6d_pose_estimation = True
            while not self.synchronization.end_6d_pose_estimation:
                pass
            self.pose_6d_resource.value = self.shared_variables.pose_6d
            self.mqtt_client.publish(
                self.pose_6d_resource.topic,
                self.pose_6d_resource,
                self.pose_6d_resource.qos,
                self.pose_6d_resource.retained
            )
            self.synchronization.start_6d_pose_estimation = False
            flag.reset_flag()
        print(flag.get_flag())



