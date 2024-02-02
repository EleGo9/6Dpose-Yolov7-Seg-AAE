import sys
from time import time, sleep

sys.path.append("src/services/thread")
from base_thread import BaseThread
sys.path.append("src/resources/models")
from base_mqtt_data_model import BaseMqttDataModel
sys.path.append("src/utils/senml/basic")
from basic_senml_record import BasicSenMLRecord
from basic_senml_pack import BasicSenMLPack


class MqttThread(BaseThread):
    def __init__(self, mqtt_publisher_client, interval_s, topics):
        super().__init__()
        self.mqtt_publisher_client = mqtt_publisher_client
        self.interval_s = interval_s
        self.topics = topics

        self.start_stop_cycle_resource = None
        self.selected_recipe_resource = None
        self.done_bags_resources = None
        self.current_object_resources = None
        self.cobot_state_cycle_resource = None
        self.safety_resource = None
        self.resources = []

        self.initialize()

    def initialize(self):
        self.add_resources()
        self.mqtt_publisher_client.connect()
        self.mqtt_publisher_client.start()
        self.mqtt_publisher_client.publish(
            "info",
            self.topics,
            0,
            True
        )

    def run(self):
        while True:
            sleep(self.interval_s)
            resources_value = [
                "In produzione" if self.shared_variables.start_stop_cycle else "Fermo",
                self.shared_variables.selected_recipe,
                str(self.shared_variables.done_bags),
                self.shared_variables.current_object,
                self.shared_variables.cobot_state_cycle,
                self.shared_variables.safety
            ]
            for r, rv in zip(self.resources, resources_value):
                r.value = rv
                record = BasicSenMLRecord(
                    r.name, int(time()*1000), "", r.value
                )
                pack = BasicSenMLPack()
                pack.append(record)
                self.mqtt_publisher_client.publish(
                    r.topic,
                    pack,
                    r.qos,
                    r.retained
                )

    def add_resources(self):
        self.start_stop_cycle_resource = BaseMqttDataModel(
            "start_stop_cycle",
            self.topics[0],
            0,
            False
        )
        self.selected_recipe_resource = BaseMqttDataModel(
            "selected_recipe",
            self.topics[1],
            0,
            False
        )
        self.done_bags_resources = BaseMqttDataModel(
            "done_bags",
            self.topics[2],
            0,
            False
        )
        self.current_object_resources = BaseMqttDataModel(
            "current_object",
            self.topics[3],
            0,
            False
        )
        self.cobot_state_cycle_resource = BaseMqttDataModel(
            "cobot_state_cycle",
            self.topics[4],
            0,
            False
        )
        self.safety_resource = BaseMqttDataModel(
            "safety",
            self.topics[5],
            0,
            False
        )
        self.resources.append(self.start_stop_cycle_resource)
        self.resources.append(self.selected_recipe_resource)
        self.resources.append(self.done_bags_resources)
        self.resources.append(self.current_object_resources)
        self.resources.append(self.cobot_state_cycle_resource)
        self.resources.append(self.safety_resource)
