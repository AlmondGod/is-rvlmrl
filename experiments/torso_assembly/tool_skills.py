import numpy as np
from scipy.spatial.transform import Rotation
from experiments.arm_corner.environment import down_rotation
from .environment import SIZES,CORNER_PAIRS,PANEL_SIZE
class Tool:
 def __init__(self,robot,planner,kind,index,est,panel=None):
  self.targets=robot['tcp'].copy();self.rotations=robot['rotation'].copy();self.gripper=np.zeros(2);self.planner=planner;self.kind=kind;self.index=index;self.events=[];self.time=0.;self.settled=0.;self.state='route';self.failure=None
  if any(e is None for e in est):raise ValueError('Tool targeting requires current visual member estimates')
  if kind=='frame':
   h,v=CORNER_PAIRS[index];sign=-1 if index in (0,1) else 1
   e=est[v];yaw=e['yaw'];axis=np.array([-np.sin(yaw),np.cos(yaw),0.])
   if axis[1]<0:axis=-axis
   self.tip=np.array(e['center'])+sign*axis*(SIZES[v,1]/2-.018);self.tip[2]+=SIZES[v,2]/2
   direction=sign*.8*axis+np.array([0,0,-.6]);x=np.cross(direction,[0,0,1.]);x/=np.linalg.norm(x);self.R=np.column_stack((x,np.cross(direction,x),direction))
  else:
   if panel is None:raise ValueError('Panel fastening requires an observed sheet pose')
   n=np.array(panel['rotation'])[:,2];self.tip=np.array(est[index]['center']).copy()
   if abs(n[2])<.9:raise ValueError('Panel not sufficiently horizontal for planned fastener')
   self.tip[2]=panel['center'][2]+(PANEL_SIZE[2]/2+.003-n[0]*(self.tip[0]-panel['center'][0])-n[1]*(self.tip[1]-panel['center'][1]))/n[2];self.R=down_rotation(0)
  baseR=self.R.copy();self.route=None;failures=[]
  # Tool roll is free for a circular nail axis. Use that physical freedom
  # to clear the two parked arms without changing the intended nail ray.
  for roll in (0.,np.pi/2,np.pi,-np.pi/2):
   self.R=baseR@Rotation.from_rotvec([0,0,roll]).as_matrix()
   try:self.route=np.array(planner.plan(2,robot,est,self.tip-self.R[:,2]*.10,self.R,panel));break
   except (RuntimeError,ValueError) as ex:failures.append(str(ex))
  if self.route is None:raise RuntimeError('No collision-free nailer roll: '+str(failures))
  self.ix=1
 def command(self,r):
  self.time+=.02;override=None
  if self.state in ('route','return'):
   override={2:self.route[self.ix]}
   if np.max(abs(r['joints'][2]-self.route[self.ix]))<.012:
    self.ix+=1 if self.state=='route' else -1
    if self.state=='route' and self.ix==len(self.route):self.state='seat';self.time=0.;override=None;self.targets[2]=self.tip-self.R[:,2]*(.004 if self.kind=='frame' else 0.);self.rotations[2]=self.R
    elif self.state=='return' and self.ix<0:self.state='done';override=None;self.targets[2]=r['tcp'][2].copy();self.rotations[2]=r['rotation'][2].copy()
  elif self.state=='seat':
   good=np.linalg.norm(r['tcp'][2]-self.tip)<.005 and np.linalg.norm(Rotation.from_matrix(self.R@r['rotation'][2].T).as_rotvec())<.04
   self.settled=self.settled+.02 if good else 0.
   if self.settled>.5:self.events.append({'kind':self.kind,'index':self.index,'time':r['time'],'tcp':r['tcp'][2].tolist(),'abstract':True,'physical_bond':False});self.targets[2]=self.tip-self.R[:,2]*.10;self.state='retract';self.time=0.
  elif self.state=='retract':
   if np.linalg.norm(r['tcp'][2]-self.targets[2])<.006:self.state='return';self.ix=len(self.route)-1;self.time=0.
  if self.time>45:raise RuntimeError('tool_timeout:'+self.state)
  return self.targets.copy(),self.rotations.copy(),self.gripper.copy(),override
