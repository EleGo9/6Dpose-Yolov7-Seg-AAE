import sys


sys.path.append("src/services/modbus")
from modbus_client import ModbusClient

if __name__ == "__main__":
    ip_address = "192.168.1.100"
    port = 502
    modbus_client = ModbusClient(ip_address, port)

    print(modbus_client.read_holding_register(135))
    modbus_client.write_holding_register(129, 17)

    for i in range(6):
        print(modbus_client.decode_read_holding_registers(400+i))
    modbus_client.encode_write_holding_registers(129, -1340)

