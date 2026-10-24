"""Lossless trusted-local pickle codec deduplicating embedded MuJoCo models.

MuJoCo's native MjData pickle contains a full serialized MjModel. Several data
objects can therefore repeat a large identical model within one environment.
This codec interns identical 64 KiB model blocks in the pickle memo, including
shared blocks across slightly different collision-planning model variants. It does
not approximate integration state or share mutable model instances on restore.
Only use with locally generated snapshots; pickle executes Python objects.
"""
import hashlib
import pickle


def _model(state):
    import mujoco
    value=mujoco.MjModel.__new__(mujoco.MjModel)
    value.__setstate__(b''.join(state))
    return value


def _data(prefix,model,suffix):
    import mujoco
    value=mujoco.MjData.__new__(mujoco.MjData)
    value.__setstate__(prefix+b''.join(model)+suffix)
    return value


class CompactPickler(pickle.Pickler):
    def __init__(self,file,protocol=5):
        super().__init__(file,protocol=protocol)
        self.models={}
        self.blocks={}
    def intern(self,state):
        digest=hashlib.sha256(state).digest()
        if digest in self.models:
            value=self.models[digest]
            if b''.join(value)!=state:raise ValueError('Model digest collision')
            return value
        chunks=[]
        for offset in range(0,len(state),65536):
            chunk=state[offset:offset+65536]
            key=hashlib.sha256(chunk).digest()
            shared=self.blocks.setdefault(key,chunk)
            if shared!=chunk:raise ValueError('Model block digest collision')
            chunks.append(shared)
        value=tuple(chunks);self.models[digest]=value
        return value
    def reducer_override(self,obj):
        import mujoco
        if type(obj) is mujoco.MjModel:
            return _model,(self.intern(obj.__getstate__()),)
        if type(obj) is mujoco.MjData:
            state=obj.__getstate__();model=obj.model.__getstate__()
            # Fail closed if this native serialization layout changes. No
            # offsets inside the model or dynamic state are interpreted.
            if not state.startswith(model,1):
                raise ValueError('Unrecognized MuJoCo data serialization')
            return _data,(state[:1],self.intern(model),state[1+len(model):])
        return NotImplemented


def dump(obj,file,protocol=5):
    CompactPickler(file,protocol).dump(obj)


def load(file):
    return pickle.load(file)
