import sys

sys.path.append("src/services/thread")
from base_thread import BaseThread
sys.path.append("src/utils/cycle/production")
from production_cycle import ProductionCycle
sys.path.append("src/utils/trigger")
from trigger import Trigger
sys.path.append("src/utils/timer")
from timer import Timer


class ModbusThread(BaseThread):
    def __init__(self, modbus_client, modbus_client_registers_address_configuration, production_cycle_states_configuration):
        super().__init__()
        self.modbus_client = modbus_client
        self.modbus_client_registers_address_configuration = modbus_client_registers_address_configuration
        self.production_cycle_states_configuration = production_cycle_states_configuration

        self.production_cycle = None
        self.start_cycle_trigger = None
        self.joints_current_consumption_timer = None
        self.joints_temperature_timer = None
        self.system_general_info_timer = None

        self.retained_start_cycle = False

        self.cobot_state = 0
        self.return_state = 0

        self.initialize()

    def initialize(self):
        self.production_cycle = ProductionCycle(self.shared_variables)
        self.start_cycle_trigger = Trigger()
        self.joints_current_consumption_timer = Timer(1000)
        self.joints_temperature_timer = Timer(1000)
        self.system_general_info_timer = Timer(250)
        self.joints_current_consumption_timer.set()
        self.joints_temperature_timer.set()
        self.system_general_info_timer.set()

    def run(self):
        while True:
            self.read_cobot_state()
            self.retained_start_cycle = self.production_cycle.start()
            self.start_cycle_trig()
            if self.retained_start_cycle:
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
            self.joints_current_consumption()
            self.joints_temperature()
            self.system_general_info()

    def read_cobot_state(self):
        value = self.modbus_client.read_holding_register(
            self.modbus_client_registers_address_configuration.COBOT_STATE
        )
        self.cobot_state = value

    def set_cycle(self):
        self.return_state = self.cobot_state
        end_effector_index = self.production_cycle.get_object_type_end_effector_index() # randint(0, 2)
        box_position = self.production_cycle.get_object_type_box_position() # [5120, -320, 2650, 0, 31415, 0]
        self.modbus_client.write_holding_register(
            self.modbus_client_registers_address_configuration.END_EFFECTOR_INDEX,
            end_effector_index
        )
        for i, v in enumerate(box_position):
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
            pose_6d = self.shared_variables.pose_6d
            for i, v in enumerate(pose_6d):
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
        self.production_cycle.control()
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

    def start_cycle_trig(self):
        if self.start_cycle_trigger.trig(self.retained_start_cycle):
            self.cobot_state = 0
            self.reset_return_state()

    def joints_current_consumption(self):
        if self.joints_current_consumption_timer.expired():
            for i in range(6):
                self.shared_variables.joints_current_consumption_ma[i] = abs(self.modbus_client.decode_read_holding_registers(290 + i))

    def joints_temperature(self):
        if self.joints_temperature_timer.expired():
            for i in range(6):
                self.shared_variables.joints_temperature_c[i] = self.modbus_client.decode_read_holding_registers(300 + i)

    def joints_position(self):
        for i in range(6):
            v = self.modbus_client.decode_read_holding_registers(400+i)

    def system_general_info(self):
        if self.system_general_info_timer.expired():
            if self.cobot_state != 0:
                self.shared_variables.cobot_state_cycle = self.production_cycle_states_configuration.STATES[self.cobot_state//10]
            safety = self.modbus_client.read_holding_register(
                self.modbus_client_registers_address_configuration.EMERGENCY_STOPPED
            )
            self.shared_variables.safety = "Emergenza" if safety else "In sicurezza"
