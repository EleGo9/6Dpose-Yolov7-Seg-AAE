import sys
from time import time

sys.path.append("src/utils/format")
from interface_format import IFormat

sys.path.append("src/utils/senml")
from senml_pack import SenMLPack
from senml_record import SenMLRecord

sys.path.append("src/resources/models")
from resource_model import ResourceModel


class FormatJsonSenML(IFormat):
    def __init__(self):
        pass

    def initialize(self):
        pass

    def to_format(self, resource: ResourceModel):
        senml_pack = SenMLPack()

        senml_record = SenMLRecord()
        senml_record.set_bn(resource.get_uuid())
        senml_record.set_bver(resource.get_version())
        senml_record.set_bt(int(time()*1000))
        if (
                type(resource.get_value()) == int or
                type(resource.get_value()) == bool or
                type(resource.get_value()) == float or
                type(resource.get_value()) == str or
                (type(resource.get_value()) == list and len(resource.get_value()) <= 1)
        ):
            self.set_senml_record_value(senml_record, resource.get_value())
            senml_record.set_bu(resource.get_unit()[0])
            senml_pack.add_senml_record_object(senml_record)
        elif type(resource.get_value()) == list:
            senml_record.set_n(resource.get_name()[0])
            self.set_senml_record_value(senml_record, resource.get_value()[0])
            if len(resource.get_unit()) == 1:
                senml_record.set_bu(resource.get_unit()[0])
            else:
                senml_record.set_u(resource.get_unit()[0])
            senml_pack.add_senml_record_object(senml_record)
            for i in range(1, len(resource.get_value())):
                senml_record = SenMLRecord()
                if len(resource.get_name()) == 1:
                    senml_record.set_n(resource.get_name()[0])
                else:
                    senml_record.set_n(resource.get_name()[i])
                self.set_senml_record_value(senml_record, resource.get_value()[i])
                if len(resource.get_unit()) != 1:
                    senml_record.set_u(resource.get_unit()[i])
                senml_pack.add_senml_record_object(senml_record)
        return senml_pack

    def from_format(self, senml_pack, resource: ResourceModel):
        senml_pack = senml_pack.get_senml_pack()
        if len(senml_pack) == 1:
            v = senml_pack[0].get_v()
            vb = senml_pack[0].get_vb()
            vs = senml_pack[0].get_vs()
            if v is not None:
                resource.set_value(v)
            if vb is not None:
                resource.set_value(vb)
            if vs is not None:
                resource.set_value(vs)
        else:
            resource.set_value([])
            for p in senml_pack:
                v = p.get_v()
                vb = p.get_vb()
                vs = p.get_vs()
                if v is not None:
                    resource.get_value().append(p.get_v())
                if vb is not None:
                    resource.get_value().append(p.get_vb())
                if vs is not None:
                    resource.get_value().append(p.get_vs())

    @staticmethod
    def set_senml_record_value(senml_record, value):
        if type(value) == int or type(value) == float:
            senml_record.set_v(value)
            senml_record.set_vb(None)
            senml_record.set_vs(None)
        elif type(value) == list:
            if len(value) == 0:
                senml_record.set_v(None)
            elif len(value) == 1:
                senml_record.set_v(value[0])
            else:
                senml_record.set_v(value)
            senml_record.set_vb(None)
            senml_record.set_vs(None)
        elif type(value) == bool:
            senml_record.set_v(None)
            senml_record.set_vb(value)
            senml_record.set_vs(None)
        elif type(value) == str:
            senml_record.set_v(None)
            senml_record.set_vb(None)
            senml_record.set_vs(value)
