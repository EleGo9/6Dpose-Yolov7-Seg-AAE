import sys
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.payload import BinaryPayloadBuilder

sys.path.append("src/services/modbus")
from interface_modbus_client import IModbusClient


class ModbusClient(IModbusClient):
    def __init__(self, ip_address, port):
        self.ip_address = ip_address
        self.port = port
        self.modbus_tcp_client = None
        self.initialize()

    def initialize(self):
        self.modbus_tcp_client = ModbusTcpClient(self.ip_address, self.port)

    def read_holding_register(self, modbus_address):
        value = self.modbus_tcp_client.read_holding_registers(modbus_address, 1, unit=1).registers[0]
        return value

    def read_holding_registers(self, modbus_address, count):
        values = self.modbus_tcp_client.read_holding_registers(modbus_address, count, unit=1).registers
        return values

    def write_holding_register(self, modbus_address, value):
        self.modbus_tcp_client.write_register(modbus_address, value, unit=1)

    def write_holding_registers(self, modbus_address, values, count):
        self.modbus_tcp_client.write_registers(modbus_address, values, count=count, unit=1, skip_encode=True)

    def encode_write_holding_registers(self, modbus_address, value):
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)
        builder.add_16bit_int(value)
        payload = builder.to_registers()
        payload = builder.build()
        self.write_holding_registers(modbus_address, payload, 1)

    def decode_read_holding_registers(self, modbus_address):
        payload = self.modbus_tcp_client.read_holding_registers(modbus_address, 1, unit=1)
        decoder = BinaryPayloadDecoder.fromRegisters(payload.registers, byteorder=Endian.Big, wordorder=Endian.Big)
        values = decoder.decode_16bit_int()
        return values
