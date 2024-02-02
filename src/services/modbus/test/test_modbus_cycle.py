import sys
import time

from pymodbus.client.sync import ModbusTcpClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.payload import BinaryPayloadBuilder

from time import sleep
from random import randint
from scipy.spatial.transform import Rotation as R

"""
FSM UR5:
    - 10: home
    - 20: new cycle (mb)
    - 30: release end effector
    - 40: catch end effector
    - 50: take photo (mb)
    - 60: approach pick
    - 70: pick
    - 80: place
"""

data = {"INDEX_GRIPPER": {"ADDRESS": 128, "DATA": 0},
        "POS_PLACE": {"ADDRESS": 129, "DATA": [0, 0, 0, 0, 0, 0]},
        "BEGIN_COM": {"ADDRESS": 137, "DATA": 1},
        "STATUS_ROBOT": {"ADDRESS": 135, "DATA": 0},
        "STATUS_PC": {"ADDRESS": 136, "DATA": 0}}

client = None

old_state = 0
counter = 0
SET_CYCLE = 30
TAKE_PHOTO = 120
CONFIRM_PLACE = 190


def main():
    global client
    client = ModbusTcpClient("192.168.1.100", port=502)

    while True:
        connect_robot()


def connect_robot():
    global client, data, old_state, counter

    data["STATUS_ROBOT"]["DATA"] = client.read_holding_registers(data["STATUS_ROBOT"]["ADDRESS"], 1, unit=1).registers[0]
    if old_state != data["STATUS_ROBOT"]["DATA"]:
        old_state = data["STATUS_ROBOT"]["DATA"]

    if data["STATUS_ROBOT"]["DATA"] == SET_CYCLE and data["STATUS_PC"]["DATA"] == 0:
        data["STATUS_PC"]["DATA"] = data["STATUS_ROBOT"]["DATA"]
        index = counter
        index = randint(0, 2)
        position = [5120, -320, 2650, 0, 31415, 0] #15708, 31415-8000, 8000]
        client.write_register(data["INDEX_GRIPPER"]["ADDRESS"], index, unit=1)
        encoded_position(data["POS_PLACE"]["ADDRESS"], position)
        client.write_register(data["STATUS_PC"]["ADDRESS"], data["STATUS_PC"]["DATA"], unit=1)
        print()
        print("write index and pose")
        print(index)
        print(position)
        counter += 1
        counter %= 3

    elif data["STATUS_ROBOT"]["DATA"] == TAKE_PHOTO and data["STATUS_PC"]["DATA"] == 0:
        data["STATUS_PC"]["DATA"] = data["STATUS_ROBOT"]["DATA"]
        time.sleep(1)
        position = [6120, -1340, 700, 8000, 31415, 0]
        encoded_position(data["POS_PLACE"]["ADDRESS"], position)
        client.write_register(data["STATUS_PC"]["ADDRESS"], data["STATUS_PC"]["DATA"], unit=1)
        print("write 6d pose")
        print(position)

    elif data["STATUS_ROBOT"]["DATA"] == CONFIRM_PLACE and data["STATUS_PC"]["DATA"] == 0:
        data["STATUS_PC"]["DATA"] = data["STATUS_ROBOT"]["DATA"]
        client.write_register(data["STATUS_PC"]["ADDRESS"], data["STATUS_PC"]["DATA"], unit=1)
        print("place")

    elif data["STATUS_ROBOT"]["DATA"] == 0 and data["STATUS_PC"]["DATA"] != 0:
        data["STATUS_PC"]["DATA"] = data["STATUS_ROBOT"]["DATA"]
        client.write_register(data["STATUS_PC"]["ADDRESS"], data["STATUS_PC"]["DATA"], unit=1)
        print("reset pc state")

    else:
        pass


def encoded_position(mb_base_address, pos):
    global client
    for i in range(6):
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)
        builder.add_16bit_int(pos[i])
        payload = builder.to_registers()
        payload = builder.build()
        client.write_registers(mb_base_address + i, payload, unit=1, skip_encode=True)


if __name__ == "__main__":
    main()
