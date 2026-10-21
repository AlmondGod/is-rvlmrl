"""Image-initialized, encoder-controlled two-gripper retention test.
Not an Astra policy. No workpiece ground truth; hidden slip remains unobserved.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .environment import OFFSETS,SIDE_R

class Flip:
 def __init__(self,robot,planner,est,panel,grasp="sheet",smooth=False,clearance=False,outboard=False):
  if panel is None or any(x is None for x in est):raise ValueError('Fresh visual poses required')
  self.grasp=grasp;self.smooth=smooth;self.clearance=clearance;self.far_clear=False;self.outboard=outboard
  self.grasp_points=np.array(panel['center'])+OFFSETS
  if grasp=='stack':
   # Known stock thickness + RGB side-member centers. No runtime object truth.
   for arm,piece in enumerate((1,3)):
    self.grasp_points[arm]=np.array(est[piece]['center'])
    self.grasp_points[arm,2]=(float(est[piece]['center'][2])-.032+float(panel['center'][2])+.006)/2
  self.planner=planner;self.est=est;self.panel=panel;self.targets=np.array(robot['tcp']).copy();self.rotations=np.array(robot['rotation']).copy();self.gripper=np.zeros(2);self.arm=0;self.time=0.;self.angle=0.;self.state='prepare';self._plan(robot)
 def _plan(self,r):
  p=self.grasp_points[self.arm].copy();p[0]+=(-.12 if self.arm==0 else .12)
  plan=self.planner.plan_flip_approach if self.outboard else self.planner.plan
  self.route=np.array(plan(self.arm,r,self.est,p,SIDE_R[self.arm],self.panel));self.ix=1
 def switch(self,s):self.state=s;self.time=0.
 def command(self,r):
  self.time+=.02;tcp=np.array(r['tcp']);override=None
  if self.state=='prepare':
   override={self.arm:self.route[self.ix]}
   if max(abs(np.array(r['joints'][self.arm])-self.route[self.ix]))<.01:
    self.ix+=1
    if self.ix>=len(self.route):
     self.targets[self.arm]=tcp[self.arm];self.rotations[self.arm]=r['rotation'][self.arm]
     if self.arm==0:self.arm=1;self._plan(r);self.time=0.
     else:self.switch('insert');override=None
  elif self.state=='insert':
   self.targets[:2]=self.grasp_points;self.rotations[:2]=SIDE_R
   if max(np.linalg.norm(self.targets[:2]-tcp[:2],axis=1))<.003:self.switch('close')
  elif self.state=='close':
   self.gripper[:]=255
   if self.time>2.:
    self.anchor=tcp[:2].copy();self.baseR=np.array(r['rotation'][:2]).copy();self.switch('clear_lift' if self.clearance else 'lift')
  elif self.state in ('clear_lift','clear_forward','clear_far'):
   duration=8. if self.state=='clear_lift' else 12.
   u=min(1.,self.time/duration);u=u*u*u*(10-15*u+6*u*u)
   delta=np.array([0.,0.,.06]) if self.state=='clear_lift' else np.array([0.,-.20 if self.state=='clear_far' else -.10,0.])
   self.targets[:2]=self.anchor+u*delta
   if self.time>=duration and max(np.linalg.norm(self.targets[:2]-tcp[:2],axis=1))<.012:
    self.anchor=tcp[:2].copy()
    if self.state=='clear_far':self.far_clear=True
    self.switch('clear_forward' if self.state=='clear_lift' else 'lift')
  elif self.state=='lift':
   height=.40 if self.clearance and not self.far_clear else .75;dz=height-np.mean(self.anchor[:,2])
   if self.smooth:
    duration=15.;u=min(1.,self.time/duration);u=u*u*u*(10-15*u+6*u*u)
    self.targets[:2]=self.anchor+[0,0,dz*u]
   else:self.targets[:2]=self.anchor+[0,0,dz]
   # These are clearance waypoints, not placement targets. Loaded arm
   # compliance is allowed within the verified >40 mm clearance budget.
   if max(abs(tcp[:2,2]-height))<(.015 if self.clearance else .005):
    if self.clearance and not self.far_clear:self.anchor=tcp[:2].copy();self.switch('clear_far')
    else:self.pivot=tcp[:2].copy();self.switch('rotate')
  elif self.state=='rotate':
   # Opposed grips lie on the X rotation axis. Rotate wrists together,
   # advancing only while both encoders track their orientation targets.
   err=max(Rotation.from_matrix(a@b.T).magnitude() for a,b in zip(self.rotations[:2],r['rotation'][:2]))
   if err<.025:self.angle=min(np.pi,self.angle+.12*.02)
   R=Rotation.from_rotvec([self.angle,0,0]).as_matrix();self.rotations[:2]=np.array([R@x for x in self.baseR]);self.targets[:2]=self.pivot
   if self.angle>=np.pi and err<.025:self.switch('hold_inverted')
  elif self.state=='hold_inverted':
   if self.time>5:self.switch('done')
  if self.time>60:raise RuntimeError('flip_timeout:'+self.state)
  return self.targets.copy(),self.rotations.copy(),self.gripper.copy(),override
