import sys

sys.path.append("src/services/thread")
from base_thread import BaseThread
from random import randint
from time import time
start_time = 0


class ModbusThread(BaseThread):
    def __init__(self, modbus_client, modbus_client_registers_address_configuration, production_cycle_states_configuration):
        super().__init__()
        self.modbus_client = modbus_client
        self.modbus_client_registers_address_configuration = modbus_client_registers_address_configuration
        self.production_cycle_states_configuration = production_cycle_states_configuration

        self.end_effector_index = 0
        self.pose_6d = [0, 0, 0, 0, 0, 0]
        self.cobot_state = 0
        self.return_state = 0

    def initialize(self):
        pass

    def run(self):
        while True:
            self.read_cobot_state()
            if self.return_state == 0:
                if self.cobot_state == self.production_cycle_states_configuration.SET_CYCLE:
                    self.set_cycle()
                elif self.cobot_state == self.production_cycle_states_configuration.TAKE_PHOTO:
                    self.take_photo()
                elif self.cobot_state == self.production_cycle_states_configuration.CONFIRM_PLACE:
                    self.confirm_place()
                else:
                    pass
            else:
                if self.cobot_state == 0:
                    self.reset_return_state()
            self.read_joint_position_tmm()

    def read_cobot_state(self):
        value = self.modbus_client.read_holding_register(
            self.modbus_client_registers_address_configuration.COBOT_STATE
        )
        self.cobot_state = value

    def set_cycle(self):
        self.return_state = self.cobot_state
        self.end_effector_index = randint(0, 2)
        self.pose_6d = [5120, -320, 2650, 0, 31415, 0]
        self.modbus_client.write_holding_register(
            self.modbus_client_registers_address_configuration.END_EFFECTOR_INDEX,
            self.end_effector_index
        )
        for i, v in enumerate(self.pose_6d):
            self.modbus_client.encode_write_holding_registers(
                self.modbus_client_registers_address_configuration.POSE_6D + i,
                v
            )
        self.modbus_client.write_holding_register(
            self.modbus_client_registers_address_configuration.RETURN_STATE,
            self.return_state
        )

    def take_photo(self):
        self.synchronization.start_6d_pose_estimation = True
        if self.synchronization.end_6d_pose_estimation:
            self.return_state = self.cobot_state
            self.pose_6d = self.shared_variables.pose_6d
            for i, v in enumerate(self.pose_6d):
                self.modbus_client.encode_write_holding_registers(
                    self.modbus_client_registers_address_configuration.POSE_6D + i,
                    v
                )
            self.modbus_client.write_holding_register(
                self.modbus_client_registers_address_configuration.RETURN_STATE,
                self.return_state
            )
            self.synchronization.start_6d_pose_estimation = False

    def confirm_place(self):
        self.return_state = self.cobot_state
        self.modbus_client.write_holding_register(
            self.modbus_client_registers_address_configuration.RETURN_STATE,
            self.return_state
        )

    def reset_return_state(self):
        self.return_state = self.cobot_state
        self.modbus_client.write_holding_register(
            self.modbus_client_registers_address_configuration.RETURN_STATE,
            self.return_state
        )

    def read_joint_position_tmm(self):
        global start_time
        if time()*1000 >= start_time+1000:
            start_time = time()*1000
            for i in range(6):
                v = self.modbus_client.decode_read_holding_registers(400+i)
                print(v, end=" ")
            print()

    def read_joint_current_ma(self):
        pass

    def read_joint_temperature_c(self):
        pass
