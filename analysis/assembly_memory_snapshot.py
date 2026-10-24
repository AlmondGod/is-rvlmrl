"""Experimental complete in-memory copies with one copy per MuJoCo model.

Python pickle normally repeats a whole MjModel inside each MjData payload. This
trusted local codec copies distinct models once and uses MuJoCo's mj_copyData
for their data. Models are never shared with the live simulation. It is not a
portable checkpoint, reward cache, or reduced integration-state snapshot.
"""
import copy
import ctypes
import io
import pickle
from pathlib import Path

import mujoco

_LIBRARY = None


def native_copy():
    global _LIBRARY
    if _LIBRARY is None:
        package = Path(mujoco.__file__).parent
        candidates = [package/f'libmujoco.so.{mujoco.__version__}',
                      package/f'libmujoco.{mujoco.__version__}.dylib']
        libraries = [p for p in candidates if p.is_file()]
        if len(libraries) != 1:
            raise RuntimeError('Expected one installed MuJoCo runtime library')
        library = ctypes.CDLL(str(libraries[0]))
        library.mj_copyData.argtypes = [ctypes.c_void_p]*3
        library.mj_copyData.restype = ctypes.c_void_p
        _LIBRARY = library
    return _LIBRARY.mj_copyData


def clone(value):
    """Copy the entire reachable Python/native state, preserving native aliases."""
    models, data = {}, {}

    def model_copy(model):
        key = id(model)
        if key not in models:
            models[key] = copy.copy(model)
        return key

    class Writer(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, mujoco.MjModel):
                return ('model', model_copy(obj))
            if isinstance(obj, mujoco.MjData):
                key = id(obj)
                if key not in data:
                    model = models[model_copy(obj.model)]
                    destination = mujoco.MjData(model)
                    address = native_copy()(destination._address, model._address, obj._address)
                    if address != destination._address:
                        raise RuntimeError('MuJoCo data copy failed')
                    data[key] = destination
                return ('data', key)
            return None

    class Reader(pickle.Unpickler):
        def persistent_load(self, reference):
            kind, key = reference
            if kind == 'model':return models[key]
            if kind == 'data':return data[key]
            raise ValueError('Unknown native object reference')

    stream = io.BytesIO()
    Writer(stream, protocol=5).dump(value)
    size = stream.tell();stream.seek(0)
    result = Reader(stream).load()
    receipt = dict(models=len(models), data=len(data), python_payload_bytes=size,
                   method='distinct_model_copy_and_mj_copyData_v1')
    return result, receipt


class ProbeCodec:
    """Same-process benchmark adapter. The marker file contains no native state."""
    def __init__(self):
        self.snapshot = None
        self.receipts = []

    def dump(self, value, stream, protocol=5):
        if self.snapshot is not None:raise RuntimeError('Probe snapshot already exists')
        self.snapshot, receipt = clone(value)
        self.receipts.append(receipt)
        pickle.dump({'in_memory_probe': True, 'receipt': receipt}, stream, protocol=protocol)

    def load(self, stream):
        marker = pickle.load(stream)
        if marker != {'in_memory_probe': True, 'receipt': self.receipts[0]}:
            raise ValueError('Unexpected local probe marker')
        if self.snapshot is None:raise RuntimeError('Native snapshot is no longer resident')
        result, receipt = clone(self.snapshot)
        self.receipts.append(receipt)
        return result
