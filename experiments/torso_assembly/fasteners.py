"""Evaluator-side finite-strength nail surrogate, SI units.
Insertion is a ray-gated instantaneous event, NOT calibrated impact/penetration.
No welds, pose changes, glue, or unbounded strength. Parameters are provisional.
"""
from dataclasses import dataclass,asdict
import numpy as np
import mujoco
from experiments.box_assembly.fastening import Frame,Connector
from .environment import SIZES,PANEL_SIZE,CORNER_PAIRS

@dataclass(frozen=True)
class NailParameters:
 length_m: float=.075
 stiffness_N_m: float=12000.
 damping_N_s_m: float=4.
 withdrawal_N: float=45.
 lateral_N: float=70.
 failure_displacement_m: float=.004
 max_standoff_m: float=.012
 max_gap_m: float=.006
 min_embed_m: float=.015
 def __post_init__(self):
  if any(not np.isfinite(v) or v<=0 for v in asdict(self).values()):raise ValueError('Positive finite SI parameters required')

def ray_interval(origin,direction,frame,size,length):
 o=frame.local(origin);v=frame.rotation.T@direction;lo=0.;hi=length
 for x,d,h in zip(o,v,np.asarray(size)/2):
  if abs(d)<1e-10:
   if abs(x)>h:return None
  else:
   a,b=sorted(((-h-x)/d,(h-x)/d));lo=max(lo,a);hi=min(hi,b)
 if hi<=lo:return None
 return lo,hi

class NailBank:
 def __init__(self,env,parameters=None,enabled=True):
  self.env=env;self.p=parameters or NailParameters();self.enabled=enabled;self.connections=[];self.events=[];self.failures=[];self.keys=set();self.damping_work_J=0.
 def frame(self,body):
  m,d=self.env.m,self.env.d;vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,vel,0)
  origin=d.xpos[body];linear=vel[3:]+np.cross(vel[:3],origin-d.xipos[body])
  return Frame(origin.copy(),d.xmat[body].reshape(3,3).copy(),linear,vel[:3].copy())
 def fire(self,kind,index):
  key=(kind,index)
  if key in self.keys:raise ValueError('No duplicate shot or automatic repair')
  self.keys.add(key);e=self.env;p=self.p;r=e.robot();origin=r['tcp'][2];axis=r['rotation'][2][:,2]
  bodies=e.woods+[e.panel];sizes=list(SIZES)+[PANEL_SIZE];hits=[]
  for body,size in zip(bodies,sizes):
   f=self.frame(body);hit=ray_interval(origin,axis,f,size,p.length_m)
   if hit is not None:hits.append((hit[0],hit[1],body,f))
  hits.sort(key=lambda x:x[0]);expected={e.woods[j] for j in CORNER_PAIRS[index]} if kind=='frame' else {e.panel,e.woods[index]}
  reason=None
  if len(hits)<2 or {h[2] for h in hits[:2]}!=expected:reason='ray_does_not_cross_intended_materials'
  elif hits[0][0]>p.max_standoff_m:reason='tool_too_far_from_surface'
  elif hits[1][0]-hits[0][1]>p.max_gap_m:reason='joint_gap_too_large'
  elif hits[1][1]-hits[1][0]<p.min_embed_m:reason='insufficient_receiver_embed'
  if reason:
   event=dict(kind=kind,index=index,time=float(e.d.time),physical_bond=False,rejection_reason=reason,parameters_provisional=True)
   self.events.append(event);return event # latent miss stays evaluator-only
  a,b=hits[:2];point=origin+axis*((a[1]+b[0])/2);fraction=min(1.,(b[1]-b[0])/.025)
  if self.enabled:self.connections.append(dict(a=a[2],b=b[2],connector=Connector.at(a[3],b[3],point),axis_a=a[3].rotation.T@axis,fraction=fraction,key=key))
  event=dict(kind=kind,index=index,time=float(e.d.time),physical_bond=self.enabled,abstract=not self.enabled,insertion_model='instantaneous ray-gated insertion; no driving dynamics',parameters_provisional=True,body_ids=[a[2],b[2]],point_m=point.tolist(),embed_m=b[1]-b[0]);self.events.append(event);return event
 def apply(self):
  e=self.env;p=self.p
  # State cannot change within this paired-force accumulation. Reuse each
  # body's frame across its incident nails; this does not change the force law.
  frames={body:self.frame(body) for body in {c[key] for c in self.connections for key in ('a','b')}}
  for c in self.connections:
   a,b=frames[c['a']],frames[c['b']];axis=a.rotation@c['axis_a'];q=c['fraction']
   was_failed=c['connector'].failed
   force,point,energy,loss=c['connector'].evaluate(a,b,axis,p.stiffness_N_m*q,p.damping_N_s_m*q,p.withdrawal_N*q,p.lateral_N*q,p.failure_displacement_m,e.m.opt.timestep)
   if c['connector'].failed and not was_failed:
    joint=c['connector'];delta=b.world(joint.anchor_b)-a.world(joint.anchor_a)
    relative=b.velocity(point)-a.velocity(point);trial=p.stiffness_N_m*q*delta+p.damping_N_s_m*q*relative
    normal=float(trial@axis);shear=float(np.linalg.norm(trial-normal*axis));util=float(np.hypot(normal/(p.withdrawal_N*q),shear/(p.lateral_N*q)))
    self.failures.append(dict(time=float(e.d.time),kind=c['key'][0],index=c['key'][1],utilization=util,displacement_m=float(np.linalg.norm(delta)),normal_force_N=normal,shear_force_N=shear,force_limit_exceeded=util>=1.,displacement_limit_exceeded=np.linalg.norm(delta)>=p.failure_displacement_m,fracture_energy_J=joint.fracture_energy_J))
   self.damping_work_J+=loss
   mujoco.mj_applyFT(e.m,e.d,force,np.zeros(3),point,c['a'],e.d.qfrc_applied)
   mujoco.mj_applyFT(e.m,e.d,-force,np.zeros(3),point,c['b'],e.d.qfrc_applied)
 def metrics(self):
  return dict(enabled=self.enabled,failure_events=list(self.failures),inserted=len(self.events),missed=sum(not x['physical_bond'] for x in self.events) if self.enabled else 0,intact=sum(not c['connector'].failed for c in self.connections),failed=sum(c['connector'].failed for c in self.connections),peak_utilization=max([c['connector'].peak_utilization for c in self.connections],default=0.),fracture_energy_J=sum(c['connector'].fracture_energy_J for c in self.connections),damping_work_J=self.damping_work_J,parameters=asdict(self.p),physical_validation=False)
