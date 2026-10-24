"""Isolated native inner-loop prototype; not a registered training backend."""
import ctypes,hashlib,inspect,json,os,pathlib,subprocess,types
import numpy as np
import mujoco


def install():
 from experiments.torso_assembly import acceleration as acceleration
 from experiments.torso_assembly import native_connectors
 from experiments.cap_rl.support_measurement import ObservedNailBank
 native_connectors.enable()
 source=pathlib.Path(__file__).with_suffix('.c');package=pathlib.Path(mujoco.__file__).parent
 header=pathlib.Path(native_connectors.__file__).parent
 libraries=list(package.glob('libmujoco.so*'))
 if len(libraries)!=1:raise RuntimeError('Expected pinned Linux MuJoCo library')
 flags=['-O3','-std=c99','-shared','-fPIC','-ffp-contract=off','-fno-builtin-pow']
 binary=source.with_name(source.stem+f'-{os.getpid()}.so')
 command=['cc',*flags,str(source),'-I'+str(header),'-I'+str(package/'include'),str(libraries[0]),'-Wl,-rpath,'+str(package),'-lm','-o',str(binary)]
 subprocess.run(command,check=True,capture_output=True)
 lib=ctypes.CDLL(str(binary));D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');I=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
 lib.substeps.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,I,ctypes.c_int,I,D,D,I,D,ctypes.c_int,ctypes.c_int,D,I,D,D,I]
 lib.substeps.restype=ctypes.c_int
 lib.connectors.argtypes=[D,D,ctypes.c_size_t];lib.connectors.restype=None
 # Recheck the exact arithmetic compiled into this larger translation unit.
 from experiments.box_assembly.fastening import Connector,Frame
 original=pathlib.Path(native_connectors.__file__).parents[1]/'box_assembly/fastening.py'
 namespace={'__name__':'experiments.box_assembly.fastening','__file__':str(original)}
 exec(compile(original.read_text(),str(original),'exec'),namespace)
 reference=namespace['Connector'].evaluate
 checked=native_connectors.compatibility_check(lib,reference=reference)
 def inner(env,ids,limits,internal):
  if type(env.nails) is not ObservedNailBank or internal!=env.nails.apply:
   raise RuntimeError('Only the frozen observed finite-strength nail hook is supported')
  bank=env.nails;cs=bank.connections;p=bank.p
  bodies=np.array(sorted({c[k] for c in cs for k in ('a','b')}),dtype=np.int32)
  lookup={b:i for i,b in enumerate(bodies)}
  indices=np.array([[lookup[c['a']],lookup[c['b']]] for c in cs],dtype=np.int32).reshape(-1)
  axes=np.array([c['axis_a'] for c in cs],dtype=np.float64).reshape(-1)
  inputs=np.zeros((len(cs),56))
  for x,c in zip(inputs,cs):
   joint=c['connector'];q=c['fraction'];x[36:39]=joint.anchor_a;x[39:42]=joint.anchor_b
   x[45:51]=p.stiffness_N_m*q,p.damping_N_s_m*q,p.withdrawal_N*q,p.lateral_N*q,p.failure_displacement_m,env.m.opt.timestep
   x[51:]=joint.failed,joint.fracture_energy_J,joint.peak_utilization,joint.stiffness_work_J,joint.previous_k
  state=np.array([env.peak,0.,env.saturation,bank.damping_work_J,bank.panel_support_z_N,0.])
  geoms=np.full(2,-1,dtype=np.int32);failures=np.empty((8,56));times=np.empty(8);order=np.empty(8,dtype=np.int32)
  code=lib.substeps(env.m._address,env.d._address,round(.02/env.m.opt.timestep),len(bodies),bodies,
   len(cs),indices,axes,inputs,np.asarray(ids,dtype=np.int32),np.asarray(limits,dtype=np.float64),len(ids),bank.panel_z_dof,state,geoms,failures,times,order)
  if code:raise RuntimeError('Native substep invariant '+str(code))
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
  env.peak=float(state[0]);env.saturation=int(state[2]);bank.damping_work_J=float(state[3]);bank.panel_support_z_N=float(state[4])
  if geoms[0]>=0:env.worst=dict(time=float(state[1]),pair=[env.m.geom(int(g)).name or str(g) for g in geoms],depth_m=env.peak)
 # Keep the complete original control update; replace only its internal loop.
 function=inspect.getsource(acceleration.step)
 begin=function.index('    for _ in range(round(.02/m.opt.timestep)):')
 end=function.index('    return errors',begin)
 modified=function[:begin]+'    _inner(self, ids, limits, internal)\n'+function[end:]
 namespace=dict(acceleration.step.__globals__,_inner=inner);exec(compile(modified,str(source), 'exec'),namespace)
 # Preserve the registered class-level step and provenance guard. Each newly
 # constructed isolated probe world uses the instrumented instance method.
 from experiments.torso_assembly.environment import Env as TorsoEnv
 from experiments.frame_assembly_v2.environment import Env as FrameEnv
 import textwrap
 frame_source=inspect.getsource(FrameEnv.step)
 if frame_source.count('super().step(targets,rotations,gripper)')!=1:raise RuntimeError('Frame step changed')
 frame_namespace=dict(FrameEnv.step.__globals__,_control=namespace['step'])
 exec(compile(textwrap.dedent(frame_source.replace('super().step(targets,rotations,gripper)','_control(self,targets,rotations,gripper)')),str(source),'exec'),frame_namespace)
 torso_source=inspect.getsource(TorsoEnv.step)
 if torso_source.count('FrameEnv.step(self,*args,**kwargs)')!=1:raise RuntimeError('Torso step changed')
 import textwrap
 torso_namespace=dict(TorsoEnv.step.__globals__,_frame_step=frame_namespace['step'])
 exec(compile(textwrap.dedent(torso_source.replace('FrameEnv.step(self,*args,**kwargs)','_frame_step(self,*args,**kwargs)')),str(source),'exec'),torso_namespace)
 original_init=TorsoEnv.__init__
 def initialize(self,*args,**kwargs):
  original_init(self,*args,**kwargs);self.step=types.MethodType(torso_namespace['step'],self)
 TorsoEnv.__init__=initialize
 globals()['_NATIVE_STEP']=torso_namespace['step']
 return dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),control_source_sha256=hashlib.sha256(function.encode()).hexdigest(),compiled_flags=flags,compatibility=checked,mujoco=mujoco.__version__,numpy=np.__version__)
