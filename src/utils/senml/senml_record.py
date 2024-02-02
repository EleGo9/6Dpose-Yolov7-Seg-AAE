import json
from collections import namedtuple


class SenMLRecord:

    def __init__(self, *args):
        self.bn = None
        self.bver = None
        self.bt = None
        self.bv = None
        self.bu = None
        self.n = None
        self.v = None
        self.u = None
        self.vb = None
        self.vs = None
        if len(args) > 0 and isinstance(args[0], dict):
            vars(self).update(args[0])

    def get_bn(self):
        return self.bn

    def get_bver(self):
        return self.bver

    def get_bt(self):
        return self.bt

    def get_bv(self):
        return self.bv

    def get_bu(self):
        return self.bu

    def get_n(self):
        return self.n

    def get_v(self):
        return self.v

    def get_u(self):
        return self.u

    def get_vb(self):
        return self.vb

    def get_vs(self):
        return self.vs

    def set_bn(self, name):
        self.bn = name

    def set_bver(self, version):
        self.bver = version

    def set_bt(self, time):
        self.bt = time

    def set_bv(self, value):
        self.bv = value

    def set_bu(self, unit):
        self.bu = unit

    def set_n(self, name):
        self.n = name

    def set_v(self, value):
        self.v = value

    def set_u(self, unit):
        self.u = unit

    def set_vb(self, vb):
        self.vb = vb

    def set_vs(self, vs):
        self.vs = vs

    def to_json(self):
        return json.dumps(
            self,
            default=lambda o: dict((key, value) for key, value in o.__dict__.items() if value is not None),
            sort_keys=False,
            allow_nan=False
        )

    def from_json(self, payload):
        obj = json.loads(str(payload).replace("\'", "\""),
                         object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

        if hasattr(obj, "bn"):
            self.bn = obj.bn
        if hasattr(obj, "bver"):
            self.bver = obj.bver
        if hasattr(obj, "bt"):
            self.bt = obj.bt
        if hasattr(obj, "bv"):
            self.bv = obj.bv
        if hasattr(obj, "bu"):
            self.bu = obj.bu
        if hasattr(obj, "n"):
            self.n = obj.n
        if hasattr(obj, "v"):
            self.v = obj.v
        if hasattr(obj, "u"):
            self.u = obj.u
        if hasattr(obj, "vb"):
            self.vb = obj.vb
        if hasattr(obj, "vs"):
            self.vs = obj.vs
