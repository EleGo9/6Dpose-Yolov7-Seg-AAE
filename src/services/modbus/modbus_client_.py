import sys
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.payload import BinaryPayloadBuilder

import time

sys.path.append("src/services/neuralnetworks")
import neural_networks

#import Web_App.web_app_flask as web_app

client = None
connection = False
start_detection_pose_estimation = False

data = {"INDEX_GRIPPER": {"ADDRESS": 128, "DATA": 0},
        "POS_PLACE": {"ADDRESS": 129, "DATA": [0, 0, 0, 0, 0, 0]},
        "BEGIN_COM": {"ADDRESS": 137, "DATA": 1},
        "STATUS_ROBOT": {"ADDRESS": 135, "DATA": 0},
        "STATUS_PC": {"ADDRESS": 136, "DATA": 0}}

robot_data = {"position": [0, 0, 0, 0, 0, 0],
              "current": [0, 0, 0, 0, 0, 0],
              "temperature": [0, 0, 0, 0, 0, 0]}

start_time = {"position": 0,
              "current": 0,
              "temperature": 0}
memo_trig = False

bag_made = 0
object_class = 0
object_to_pick = 0


def communication():
    setup()
    while True:
        # try:
        loop()
        # except:
            # print("Error occured in MODBUS communication")


def setup():
    global client, startTime
    client = ModbusTcpClient("192.168.1.100", port=502)
    if client.connect():
       begin_robot()
       startTime = time.time() * 1000


def loop():
    global client
    while client.connect():
        connect_robot()
    client.close()


def begin_robot():
    global client, data
    data["STATUS_PC"]["DATA"] = 0
    client.write_register(data["STATUS_PC"]["ADDRESS"], data["STATUS_PC"]["DATA"], unit=1)
    client.write_register(data["BEGIN_COM"]["ADDRESS"], data["BEGIN_COM"]["DATA"], unit=1)
    print("beginCommunication")


def connect_robot():
    global client, data, startTime, start_detection_pose_estimation
    global bag_made, object_class, object_to_pick

    '''read_position()
    read_current()
    read_temperature()'''

    data["STATUS_ROBOT"]["DATA"] = client.read_holding_registers(data["STATUS_ROBOT"]["ADDRESS"], 1, unit=1).registers[0]
    
    ''' if web_app.start_stop_cycle:
        if trig_cycle():
            data["STATUS_PC"]["DATA"] = 0'''
    if data["STATUS_ROBOT"]["DATA"] == 1 and not data["STATUS_PC"]["DATA"]:
        data["STATUS_PC"]["DATA"] = data["STATUS_ROBOT"]["DATA"]
        index = 0 #web_app.recipe_selected[object_class][2]
        #position = [int(p.strip()) for p in web_app.recipe_selected[object_class][3].split(",")]
        position = [5120, -320, 2650, 0, -31415, 0]
        client.write_register(data["INDEX_GRIPPER"]["ADDRESS"], index, unit=1)
        encoded_position(129, position)
        client.write_register(data["STATUS_PC"]["ADDRESS"], data["STATUS_PC"]["DATA"], unit=1)
        print("index:", index)

        #web_app.robot_state = 0
        #web_app.object_name = web_app.recipe_selected[object_class][1]

    elif data["STATUS_ROBOT"]["DATA"] == 2 and not data["STATUS_PC"]["DATA"]:
        start_detection_pose_estimation = True
        if neural_networks.end_detection_pose_estimation:
            data["STATUS_PC"]["DATA"] = data["STATUS_ROBOT"]["DATA"]
            position = neural_networks.det_pose_est_position
            encoded_position(129, position)
            client.write_register(data["STATUS_PC"]["ADDRESS"], data["STATUS_PC"]["DATA"], unit=1)
            print("position:", position)
            start_detection_pose_estimation = False

            #web_app.robot_state = 2

    elif data["STATUS_ROBOT"]["DATA"] == 3:
        pass
    elif data["STATUS_ROBOT"]["DATA"] == 4 and not data["STATUS_PC"]["DATA"]:
        data["STATUS_PC"]["DATA"] = data["STATUS_ROBOT"]["DATA"]
        client.write_register(data["STATUS_PC"]["ADDRESS"], data["STATUS_PC"]["DATA"], unit=1)
        print("Place")

        '''object_to_pick += 1
        web_app.bag_production["object"]["made"] += 1
        control_production()
        web_app.robot_state = 3'''

    elif data["STATUS_ROBOT"]["DATA"] == 0:
        data["STATUS_PC"]["DATA"] = data["STATUS_ROBOT"]["DATA"]
        client.write_register(data["STATUS_PC"]["ADDRESS"], data["STATUS_PC"]["DATA"], unit=1)
    else:
        pass
    '''elif data["STATUS_ROBOT"]["DATA"] == 0:
        data["STATUS_PC"]["DATA"] = data["STATUS_ROBOT"]["DATA"]
        client.write_register(data["STATUS_PC"]["ADDRESS"], data["STATUS_PC"]["DATA"], unit=1)'''


def encoded_position(mb_base_address, pos):
    for i in range(6):
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)
        builder.add_16bit_int(pos[i])
        payload = builder.to_registers()
        payload = builder.build()
        client.write_registers(mb_base_address+i, payload, unit=1, skip_encode=True)


def trig_cycle():
    global memo_trig
    if web_app.start_stop_cycle and not memo_trig:
        memo_trig = True
        return True
    if not web_app.start_stop_cycle:
        memo_trig = False
        return False


def control_production():
    global bag_made, object_class, object_to_pick
    if object_to_pick == web_app.recipe_selected[object_class][4]:
        object_to_pick = 0
        object_class += 1
    if object_class == len(web_app.recipe_selected):
        object_class = 0
        web_app.bag_production["object"]["made"] = 0
        web_app.bag_production["bag"]["made"] += 1
    if web_app.bag_production["bag"]["made"] == web_app.bag_production["bag"]["to_do"]:
        web_app.bag_production["object"]["made"] = 0
        web_app.start_stop_cycle = False


def read_position():
    global client, data, start_time, robot_data
    if (time.time() * 1000 - start_time["position"]) >= 500:
        start_time["position"] = time.time() * 1000
        for i in range(6):
            result = client.read_holding_registers(400+i, 1, unit=1)
            decoder = BinaryPayloadDecoder.fromRegisters(result.registers, byteorder=Endian.Big, wordorder=Endian.Big)
            robot_data["position"][i] = abs(decoder.decode_16bit_int())


def read_current():
    global client, data, start_time, robot_data
    if (time.time() * 1000 - start_time["current"]) >= 5000:
        start_time["current"] = time.time() * 1000
        for i in range(6):
            result = client.read_holding_registers(290+i, 1, unit=1)
            decoder = BinaryPayloadDecoder.fromRegisters(result.registers, byteorder=Endian.Big, wordorder=Endian.Big)
            robot_data["current"][i] = abs(decoder.decode_16bit_int())


def read_temperature():
    global client, data, start_time, robot_data
    if (time.time() * 1000 - start_time["temperature"]) >= 5000:
        start_time["temperature"] = time.time() * 1000
        for i in range(6):
            robot_data["temperature"][i] = client.read_holding_registers(300+i, 1, unit=1).registers[0] 

# uncomment if you want to run stand-alone program
# if __name__ == "__main__":
    # communication()
