import abc


class IModbusClient(abc.ABC):

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def read_holding_register(self, modbus_address):
        pass

    @abc.abstractmethod
    def read_holding_registers(self, modbus_address, count):
        pass

    @abc.abstractmethod
    def write_holding_register(self, modbus_address, value):
        pass

    @abc.abstractmethod
    def write_holding_registers(self, modbus_address, values, count):
        pass

    @abc.abstractmethod
    def encode_write_holding_registers(self, modbus_address, value):
        pass

    @abc.abstractmethod
    def decode_read_holding_registers(self, modbus_address):
        pass

