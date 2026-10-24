"""Opt-in fused native substeps for the unchanged finite-nail force model.

The enclosing Python controller, including joint overrides, remains untouched.
Unknown force hooks fall back to the reference Python loop. Runtime/source
compatibility failures are infrastructure errors, never physical reward zero.
"""
import ctypes
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np

_LIBRARY = None
_RECEIPT = None


class NativeBackendError(BaseException):
    """Bypass the episode's catch for measured skill failures."""


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_reference():
    from .fasteners import NailBank
    from .acceleration import body_frame,audit_contacts
    from experiments.cap_rl.support_measurement import ObservedNailBank
    references=[
        (NailBank.apply,'84b2e37489f74b43b83efa74c5de8b8e511cc925d5962ba455eeb06c88e95bdf'),
        (ObservedNailBank.apply,'0a45211a9150fca3226941166055e2645a8fc0c60227d5562c3ce5b2260eb51d'),
        (body_frame,'2e95900ee3c3c1d17769df1d692a8651919eb5d09d48f9dcb96217b644c1ffc8'),
        (audit_contacts,'8363e72b54e9eeee45686e5ba05f9b2bd8f0c41117f08417ec6caa09e0f43f3b')]
    for function,digest in references:
        if hashlib.sha256(inspect.getsource(function).encode()).hexdigest()!=digest:
            raise RuntimeError('Fused substeps need revalidation: '+function.__qualname__)
    return {f.__qualname__:digest for f,digest in references}


def enable():
    global _LIBRARY,_RECEIPT
    from . import native_connectors
    references=verify_reference();native_connectors.enable()
    if _LIBRARY is not None:return
    if platform.system()!='Linux':raise RuntimeError('Fused substeps currently require the validated Linux runtime')
    compiler=shutil.which('cc')
    if compiler is None:raise RuntimeError('Fused substeps require a C compiler')
    source=Path(__file__).with_name('substep_kernel.c');package=Path(mujoco.__file__).parent
    linked=package/('libmujoco.so.'+mujoco.__version__)
    if not linked.is_file():raise RuntimeError('Expected the installed MuJoCo shared library')
    flags=['-O3','-std=c99','-shared','-fPIC','-ffp-contract=off','-fno-builtin-pow']
    specification=dict(source_sha256=sha(source),connector_source_sha256=sha(source.with_name('connector_kernel.c')),
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],flags=flags,
        system=platform.system(),architecture=platform.machine(),numpy=np.__version__,mujoco=mujoco.__version__,
        mujoco_library_sha256=sha(linked),headers_sha256={str(p.relative_to(package/'include')):sha(p) for p in sorted((package/'include').rglob('*.h'))},
        references_sha256=references)
    key=hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()
    cache=Path(os.environ.get('CAP_RL_NATIVE_SUBSTEP_CACHE',str(Path(tempfile.gettempdir())/'cap-rl-native-substeps')))
    cache.mkdir(parents=True,exist_ok=True);library=cache/(key+'.so')
    with (cache/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not library.exists():
            pending=cache/(key+f'.{os.getpid()}.pending.so')
            try:
                result=subprocess.run([compiler,*flags,str(source),'-I'+str(package/'include'),str(linked),
                    '-Wl,-rpath,'+str(package),'-lm','-o',str(pending)],capture_output=True,text=True)
                if result.returncode:raise RuntimeError('Fused substep compilation failed: '+result.stderr.strip())
                pending.replace(library)
            finally:pending.unlink(missing_ok=True)
    lib=ctypes.CDLL(str(library));D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');I=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    lib.substeps.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,I,ctypes.c_int,I,D,D,I,D,ctypes.c_int,ctypes.c_int,D,I,D,D,I]
    lib.substeps.restype=ctypes.c_int
    lib.connectors.argtypes=[D,D,ctypes.c_size_t];lib.connectors.restype=None
    checked=native_connectors.compatibility_check(lib,reference=native_connectors._REFERENCE_EVALUATE)
    _LIBRARY=lib
    _RECEIPT=dict(specification,binary_sha256=sha(library),compatibility=checked,implementation='fused_native_substeps_v1')


def receipt():
    if _LIBRARY is not None:enable()
    return dict(_RECEIPT) if _RECEIPT is not None else None


def advance(env,ids,limits,internal):
    from experiments.torso_assembly.fasteners import NailBank
    from experiments.cap_rl.support_measurement import ObservedNailBank
    from experiments.box_assembly.fastening import Frame
    if _LIBRARY is None:raise NativeBackendError("Fused backend is not enabled")
    if type(getattr(env,'nails',None)) not in (NailBank,ObservedNailBank) or internal!=env.nails.apply:return False
    bank=env.nails;cs=bank.connections;p=bank.p
    if len(cs)>8:return False
    bodies=np.array(sorted({c[k] for c in cs for k in ('a','b')}),dtype=np.int32)
    lookup={b:i for i,b in enumerate(bodies)}
    indices=np.array([[lookup[c['a']],lookup[c['b']]] for c in cs],dtype=np.int32).reshape(-1)
    axes=np.array([c['axis_a'] for c in cs],dtype=np.float64).reshape(-1)
    inputs=np.zeros((len(cs),56))
    for x,c in zip(inputs,cs):
        joint=c['connector'];q=c['fraction'];x[36:39]=joint.anchor_a;x[39:42]=joint.anchor_b
        x[45:51]=p.stiffness_N_m*q,p.damping_N_s_m*q,p.withdrawal_N*q,p.lateral_N*q,p.failure_displacement_m,env.m.opt.timestep
        x[51:]=joint.failed,joint.fracture_energy_J,joint.peak_utilization,joint.stiffness_work_J,joint.previous_k
    state=np.array([env.peak,0.,env.saturation,bank.damping_work_J,getattr(bank,"panel_support_z_N",0.),0.])
    geoms=np.full(2,-1,dtype=np.int32);failures=np.empty((8,56));times=np.empty(8);order=np.empty(8,dtype=np.int32)
    code=_LIBRARY.substeps(env.m._address,env.d._address,round(.02/env.m.opt.timestep),len(bodies),bodies,
        len(cs),indices,axes,inputs,np.asarray(ids,dtype=np.int32),np.asarray(limits,dtype=np.float64),len(ids),getattr(bank,"panel_z_dof",-1),state,geoms,failures,times,order)
    if code:raise NativeBackendError('Native substep invariant '+str(code))
    for c,x in zip(cs,inputs):
        joint=c['connector'];joint.failed=bool(x[51]);joint.fracture_energy_J,joint.peak_utilization,joint.stiffness_work_J,joint.previous_k=map(float,x[52:56])
    for z in range(int(state[5])):
        c=cs[order[z]];joint=c['connector'];x=failures[z]
        a=Frame(x[:3],x[3:12].reshape(3,3),x[12:15],x[15:18]);b=Frame(x[18:21],x[21:30].reshape(3,3),x[30:33],x[33:36]);axis=x[42:45];q=c['fraction']
        point=(a.world(joint.anchor_a)+b.world(joint.anchor_b))/2
        delta=b.world(joint.anchor_b)-a.world(joint.anchor_a);relative=b.velocity(point)-a.velocity(point)
        trial=p.stiffness_N_m*q*delta+p.damping_N_s_m*q*relative
        normal=float(trial@axis);shear=float(np.linalg.norm(trial-normal*axis));util=float(np.hypot(normal/(p.withdrawal_N*q),shear/(p.lateral_N*q)))
        bank.failures.append(dict(time=float(times[z]),kind=c['key'][0],index=c['key'][1],utilization=util,displacement_m=float(np.linalg.norm(delta)),normal_force_N=normal,shear_force_N=shear,force_limit_exceeded=util>=1.,displacement_limit_exceeded=np.linalg.norm(delta)>=p.failure_displacement_m,fracture_energy_J=joint.fracture_energy_J))
    env.peak=float(state[0]);env.saturation=int(state[2]);bank.damping_work_J=float(state[3])
    if type(bank) is ObservedNailBank:bank.panel_support_z_N=float(state[4])
    if geoms[0]>=0:env.worst=dict(time=float(state[1]),pair=[env.m.geom(int(g)).name or str(g) for g in geoms],depth_m=env.peak)
    return True
