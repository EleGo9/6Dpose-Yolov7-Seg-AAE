import sys
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.payload import BinaryPayloadBuilder

from time import sleep

from scipy.spatial.transform import Rotation as R

"""
FSM UR5:
    - home
    - new cycle (mb)
    - release end effector
    - catch end effector
    - photo (mb)
    - approach pick
    - pick
    - approach place
    - place
"""

data = {"INDEX_GRIPPER": {"ADDRESS": 128, "DATA": 0},
        "POS_PLACE": {"ADDRESS": 129, "DATA": [0, 0, 0, 0, 0, 0]},
        "BEGIN_COM": {"ADDRESS": 137, "DATA": 1},
        "STATUS_ROBOT": {"ADDRESS": 135, "DATA": 0},
        "STATUS_PC": {"ADDRESS": 136, "DATA": 0}}

client = None


def main():
    global client
    client = ModbusTcpClient("192.168.1.100", port=502)

    rotation_matrix = [
        [-0.5436181, 0.73565591, 0.40409124],
        [-0.7708589, -0.62806159, 0.10637292],
        [0.33204806, -0.25367108, 0.90851256]
    ]

    rot_m = R.from_matrix(rotation_matrix)
    euler_angles = rot_m.as_euler("xyz", degrees=True)
    rotation_vector = rot_m.as_rotvec()
    rotation_vector = [round(i*10000) % 31400 for i in rotation_vector]

    #encoded_position(129, [5120, 320, 2650, 0, 0, rotation_vector[2]])
    encoded_position(129, [17, 19, 1, 7, 3, 5345])
    reading = client.read_holding_registers(138, 1, unit=1).registers[0]
    print(reading)

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
