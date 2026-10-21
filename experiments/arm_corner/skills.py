"""Object-relative motion skills. Inputs are vision estimates and robot encoders only."""
import numpy as np
from .environment import down_rotation,SIZE

class Skills:
 def __init__(self,robot):
  self.targets=robot['tcp'].copy();self.rotations=robot['rotation'].copy();self.gripper=np.zeros(2);self.state='idle';self.arm=0;self.time=0.;self.failure=None;self.holding=[False,False];self.goal=None;self.action=None
 def start(self,action,est,robot):
  if self.state not in ('idle','done','failed'):raise ValueError('Skill already busy')
  kind=action['skill'];self.arm=int(action.get('arm',0));self.time=0.;self.failure=None;self.action=action;self.targets=robot['tcp'].copy();self.rotations=robot['rotation'].copy()
  if kind in ('pick','stabilize'):
   if est[self.arm] is None:raise ValueError('No current visual estimate')
   self.state='approach';self.start_z=est[self.arm]['center'][2];self.rotations[self.arm]=down_rotation(est[self.arm]['yaw']);self.targets[self.arm]=est[self.arm]['center']+[0,0,.20]
  elif kind=='align_corner':
   if not self.holding[0] or est[0] is None or est[1] is None:raise ValueError('Need held moving member and both visible members')
   self.state='align';self.rotations[0]=down_rotation(est[1]['yaw']-np.pi/2)
  elif kind=='release':self.state='release';self.gripper[self.arm]=0
  elif kind=='stop':self.state='done'
  else:raise ValueError('Unsupported skill')
 def command(self,robot,est):
  self.time+=.02;a=self.arm;tcp=robot['tcp'];e=est[a];live=e is not None and robot['time']-e['time']<.4
  def switch(s):self.state=s;self.time=0.
  if self.state in ('approach','descend','close','probe','lift','align','lower') and not live:
   self.targets[a]=tcp[a]
   if self.time>2:self.failure='lost_visual_estimate';switch('failed')
   return self.targets,self.rotations,self.gripper
  if self.state=='approach':
   if np.linalg.norm(tcp[a]-self.targets[a])<.005:switch('descend')
   elif self.time>8:self.failure='approach_unreachable';switch('failed')
  elif self.state=='descend':
   self.targets[a,:2]=e['center'][:2];self.targets[a,2]=max(self.start_z-.014,self.targets[a,2]-.035*.02)
   if abs(tcp[a,2]-(self.start_z-.014))<.003:switch('close')
   elif self.time>10:self.failure='descent_tracking';switch('failed')
  elif self.state=='close':
   self.gripper[a]=180
   if self.time>2:
    if self.action['skill']=='stabilize':switch('done')
    else:self.probe_start=e['center'].copy();self.tcp_start=tcp[a].copy();switch('probe')
  elif self.state=='probe':
   self.targets[a,2]=min(self.tcp_start[2]+.04,self.targets[a,2]+.025*.02)
   if e['center'][2]-self.probe_start[2]>.02:self.holding[a]=True;switch('lift')
   elif self.time>3:self.failure='probe_no_visual_lift';switch('failed')
  elif self.state=='lift':
   self.targets[a,2]+=np.clip(.13-e['center'][2],-.03*.02,.03*.02)
   if abs(e['center'][2]-.13)<.005 and self.time>1:switch('done')
   elif self.time>6:self.failure='lift_timeout';switch('failed')
  elif self.state=='align':
   r=est[1]
   if r is None:self.failure='reference_not_visible';switch('failed')
   else:
    theta=r['yaw']-np.pi/2;u=np.array([-np.sin(theta),np.cos(theta),0]);ref=np.array([u[1],-u[0],0]);self.goal=r['center']+ref*(SIZE[1]/2-SIZE[0]/2)-u*(SIZE[1]/2+SIZE[0]/2+.002)
    desired=self.goal.copy();desired[2]=max(r['center'][2]+.085,.13);self.targets[0]=tcp[0]+desired-est[0]['center'];self.rotations[0]=down_rotation(theta)
    yawerr=abs(np.arctan2(np.sin(theta-est[0]['yaw']),np.cos(theta-est[0]['yaw'])))
    if np.linalg.norm(est[0]['center'][:2]-self.goal[:2])<.003 and yawerr<.02:switch('lower')
    elif self.time>12:self.failure='alignment_timeout';switch('failed')
  elif self.state=='lower':
   r=est[1];theta=r['yaw']-np.pi/2;u=np.array([-np.sin(theta),np.cos(theta),0]);ref=np.array([u[1],-u[0],0]);self.goal=r['center']+ref*(SIZE[1]/2-SIZE[0]/2)-u*(SIZE[1]/2+SIZE[0]/2+.002)
   self.targets[0,:2]=tcp[0,:2]+self.goal[:2]-est[0]['center'][:2];self.targets[0,2]-=.025*.02
   if est[0]['center'][2]-r['center'][2]<.002:switch('done')
   elif self.time>8:self.failure='lower_timeout';switch('failed')
  elif self.state=='release':
   if self.time>2:self.targets[a,2]+=.15;self.holding[a]=False;switch('retreat')
  elif self.state=='retreat':
   if self.time>4:switch('done')
  return self.targets.copy(),self.rotations.copy(),self.gripper.copy()
