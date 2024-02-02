import sys
import json

from senml_record import SenMLRecord
sys.path.append("src/error/mqtt")
from senml_format_error import SenmlFormatError


class SenMLPack:
    def __init__(self, *args):
        self._senml_pack = []
        if len(args) > 0 and isinstance(args[0], dict):
            vars(self).update(args[0])

    def get_senml_pack(self):
        return self._senml_pack

    def set_senml_pack(self, senml_pack):
        self._senml_pack = senml_pack

    def add_senml_record(self, bver=None, bn=None, bt=None, bu=None, bv=None, v=None, n=None, u=None, vb=None):
        record = SenMLRecord()
        if bver is not None:
            record.set_bver(bver)
        if bn is not None:
            record.set_bn(bn)
        if bt is not None:
            record.set_bt(bt)
        if bu is not None:
            record.set_bu(bu)
        if bv is not None:
            record.set_bv(bv)
        if v is not None:
            record.set_v(v)
        if n is not None:
            record.set_n(n)
        if u is not None:
            record.set_u(u)
        if vb is not None:
            record.set_vb(vb)
        self._senml_pack.append(record)

    def add_senml_record_object(self, record):
        if isinstance(record, SenMLRecord):
            self._senml_pack.append(record)
        else:
            raise SenmlFormatError("Object format is not support -- required senml") from None

    def to_json(self):
        return json.dumps(
            self._senml_pack,
            default=lambda o: dict((key, value) for key, value in o.__dict__.items() if value is not None),
            sort_keys=False,
            allow_nan=False
        )

    def from_json(self, message):
        self._senml_pack = []
        senml_pack = json.loads(message)
        for senml_record in senml_pack:
            senml_record = json.dumps(senml_record)
            self._senml_pack.append(json.loads(senml_record, object_hook=SenMLRecord))
