"""Privileged kinematic diagnosis only; does not produce policy evidence."""
import numpy as np,mujoco
from .environment import Env
from .planner import Planner
from .tool_skills import Tool
E=Env();rows=np.load('outputs/torso_assembly/development_resume23/raw.npz')['rows'];row=rows[-1];E.d.qpos[:]=row[1:1+E.m.nq];mujoco.mj_forward(E.m,E.d);pl=Planner(E)
est=[{'center':E.d.xpos[b].copy(),'yaw':float(np.arctan2(-E.d.xmat[b].reshape(3,3)[0,1],E.d.xmat[b].reshape(3,3)[1,1]))} for b in E.woods]
r=E.robot()
for name in ('shuttle_x','rack_slide0','rack_slide1'):r[name]=float(E.d.qpos[int(E.m.joint(name).qposadr[0])])
original=pl.valid;seen={}
def valid(q):
 ok=original(q)
 if not ok:
  for c in pl.d.contact:
   names=tuple(E.m.geom(g).name for g in (c.geom1,c.geom2))
   if c.dist<-.000005 and any(n.startswith('a2_') or n.startswith('nailer') for n in names):seen[names]=seen.get(names,0)+1
 return ok
pl.valid=valid
try:Tool(r,pl,'frame',3,est)
except Exception as ex:print('FAIL',ex)
print(sorted(seen.items(),key=lambda p:-p[1])[:15],flush=True)
