import hashlib,json,pickle,time,types
from pathlib import Path
import numpy as np
import mujoco
from experiments.torso_assembly.acceleration import configure
from experiments.torso_assembly.environment import Env
from experiments.cap_rl.support_measurement import install_observer
from experiments.box_assembly.fastening import Connector
import native_substep18 as candidate
configure('vectorized_native')
e=Env(layout_seed=51002,physical_nails=True);install_observer(e)
# Synthetic constitutive fixture only. These connections are never task rewards.
for k,(a,b) in enumerate(zip(e.woods[:3],e.woods[1:])):
 fa,fb=e.nails.frame(a),e.nails.frame(b);point=(fa.position+fb.position)/2
 e.nails.connections.append(dict(a=a,b=b,connector=Connector.at(fa,fb,point),axis_a=np.array([0.,0.,1.]),fraction=.4+.2*k,key=('frame',k)))
e.d.qvel[int(e.m.jnt_dofadr[e.m.body_jntadr[e.woods[0]]])]=.15
b=pickle.dumps(e,protocol=5);other=pickle.loads(b);del b
receipt=candidate.install();other.step=types.MethodType(candidate._NATIVE_STEP,other)
rows=[];timings=[0.,0.]
for step in range(100):
 if step==40:
  for env in (e,other):
   for c in env.nails.connections:c["connector"].anchor_a[0]+=.01
 for index,env in enumerate((e,other)):
  t=time.monotonic();env.step(env.targets.copy(),env.rotations.copy(),env.gripper.copy(),{0:env.qtarget[0].copy()} if step%3==0 else None);timings[index]+=time.monotonic()-t
 for field in ('qpos','qvel','ctrl','qacc','qacc_warmstart','xpos','xmat','qfrc_applied','actuator_force'):
  if not np.array_equal(getattr(e.d,field),getattr(other.d,field)):
   raise AssertionError((step,field,float(np.max(abs(getattr(e.d,field)-getattr(other.d,field))))))
 assert e.nails.metrics()==other.nails.metrics(),(step,'nail metrics',e.nails.metrics(),other.nails.metrics())
 assert e.nails.panel_support_z_N==other.nails.panel_support_z_N
 assert (e.peak,e.worst,e.saturation)==(other.peak,other.worst,other.saturation)
print(json.dumps(dict(passes=True,steps=100,simulated_seconds=2.,timings=timings,speedup=timings[0]/timings[1],failed_nails=len(e.nails.failures),receipt=receipt,scope='Synthetic numerical fixture only, no physical reward or episode-speed claim.')),flush=True)
