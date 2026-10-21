"""Bounded image/encoder feedback skills; no runtime object ground truth."""
import numpy as np
from .environment import PANEL_GOAL,OFFSETS,SIDE_R,HOME

class Skills:
 def __init__(self,robot,planner,frame_estimates=None):
  self.targets=np.array(robot['tcp']).copy();self.rotations=np.array(robot['rotation']).copy();self.gripper=np.zeros(2);self.state='idle';self.time=0.;self.failure=None;self.holding=False;self.released=False;self.events=[];self.planner=planner;self.frame_estimates=frame_estimates;self.rack=0.;self.overhead=0.;self.override=None;self.begin_grasp=False;self.settled=0.
 def switch(self,state):self.state=state;self.time=0.;self.settled=0.
 def start(self,kind,member,robot,estimate):
  if self.state not in ('idle','done'):raise ValueError('Skill is busy or failed')
  if estimate is None:raise ValueError('No current sheet image estimate')
  self.targets=np.array(robot['tcp']).copy();self.rotations=np.array(robot['rotation']).copy()
  if kind=='position':
   if self.holding or self.released:raise ValueError('Position already attempted')
   self.initial_height=float(estimate['center'][2]);self.clearance_lift=.018;self.rotations[:2]=SIDE_R;self.switch('approach')
  elif kind=='release':
   if not self.holding or np.linalg.norm(estimate['center'][:2])>.008:raise ValueError('Release needs a positioned held sheet')
   self.switch('release')
  elif kind=='nail':
   if not self.released or np.linalg.norm(estimate['center'][:2])>.01 or abs(estimate['yaw'])>.02:raise ValueError('Nailing needs released visually aligned sheet')
   if member not in range(4) or any(x['member']==member for x in self.events):raise ValueError('Invalid or duplicate member')
   self.member=member;self.tip,self.toolR=nail_target(member)
   if robot['carriage']>-.395:self.override={2:np.array(robot['joints'][2]).copy()};self.switch('lower_carriage')
   else:self.plan_tool(robot,estimate)
  else:raise ValueError('Unknown skill')
 def plan_retreat(self,robot,estimate):
  # Empty grippers follow a collision-checked joint route. Cartesian outward
  # increments can cross a wrist singularity on the wider shoulder layout.
  self.route=np.array(self.planner.plan(self.retreat_arm,robot,self.frame_estimates,HOME[self.retreat_arm],SIDE_R[self.retreat_arm],estimate));self.route_index=1
 def plan_tool(self,robot,estimate):
  self.route=np.array(self.planner.plan(robot,estimate,robot['carriage'],robot['rack'],self.tip+[0,0,.08],self.toolR));self.route_index=1;self.override={2:self.route[1]};self.switch('tool_route')
 def command(self,robot,estimate,visual):
  self.time+=.02;tcp=np.array(robot['tcp']);p=None if estimate is None else np.array(estimate['center']);R=None if estimate is None else np.array(estimate['rotation'])
  if self.state in ('idle','done','failed'):return self.targets,self.rotations,self.gripper,self.override
  if self.state in ('approach','insert','close','lift','withdraw_shelves','align','lower') and estimate is None:raise RuntimeError('Sheet not observed')
  if self.state=='approach':
   self.targets[:2]=p+OFFSETS;self.targets[0,0]-=.12;self.targets[1,0]+=.12
   if max(np.linalg.norm(self.targets[:2]-tcp[:2],axis=1))<.004:self.switch('insert')
  elif self.state=='insert':
   self.targets[:2]=p+OFFSETS
   if max(np.linalg.norm(self.targets[:2]-tcp[:2],axis=1))<.003:self.switch('close')
  elif self.state=='close':
   self.gripper[:]=255
   if self.time>2:
    # Compare lift against the same RGB pose used to initialize HeldSheet,
    # not the earlier approach image (occlusion can shift that estimate).
    self.initial_height=float(estimate['center'][2]);self.anchor=tcp[:2].copy();self.begin_grasp=True;self.holding=True;self.switch('lift')
  elif self.state=='lift':
   self.targets[:2]=self.anchor+[0,0,self.clearance_lift]
   if p[2]>self.initial_height+.013 and visual is not None and visual['center'][2]>self.initial_height+.010:self.switch('withdraw_shelves')
  elif self.state=='withdraw_shelves':
   self.rack=.2
   if self.time>3:self.switch('align')
  elif self.state in ('align','lower'):
   desired=np.array([0,0,self.initial_height+self.clearance_lift if self.state=='align' else PANEL_GOAL[2]+.0015]);delta=desired-p
   if self.state=='lower':delta[2]=np.clip(delta[2],-.0002,.0002)
   self.targets[:2]=p+(R.T@(tcp[:2]-p).T).T+delta;self.rotations[:2]=np.array([R.T@r for r in robot['rotation'][:2]])
   good=np.linalg.norm(p[:2])<.004 and abs(estimate['yaw'])<.005
   if self.state=='align' and good:self.switch('lower')
   elif self.state=='lower' and good and abs(p[2]-desired[2])<.001 and np.degrees(np.arccos(np.clip(R[2,2],-1,1)))<.8:self.switch('done')
  elif self.state=='release':
   self.gripper[:]=0
   if self.time>2.6:
    self.holding=False
    if self.planner is None:self.switch('retreat')
    else:self.retreat_arm=0;self.plan_retreat(robot,estimate);self.switch('retreat_route')
  elif self.state=='retreat_route':
   self.override={self.retreat_arm:self.route[self.route_index]}
   if np.max(abs(np.array(robot['joints'][self.retreat_arm])-self.route[self.route_index]))<.01:
    self.route_index+=1
    if self.route_index>=len(self.route):
     self.targets[self.retreat_arm]=tcp[self.retreat_arm];self.rotations[self.retreat_arm]=robot['rotation'][self.retreat_arm];self.override=None
     if self.retreat_arm==0:self.retreat_arm=1;self.plan_retreat(robot,estimate);self.time=0.
     else:self.released=True;self.switch('done')
  elif self.state=='retreat':
   self.targets[0,0]-=.0007;self.targets[1,0]+=.0007
   if self.time>4:self.released=True;self.switch('done')
  elif self.state=='lower_carriage':
   self.overhead=max(-.4,self.overhead-.001)
   if abs(robot['carriage']+.4)<.002 and self.time>9:self.plan_tool(robot,estimate)
  elif self.state in ('tool_route','tool_return'):
   self.override={2:self.route[self.route_index]}
   if np.max(abs(np.array(robot['joints'][2])-self.route[self.route_index]))<.015:
    if self.state=='tool_route':
     if self.route_index<len(self.route)-1:self.route_index+=1
     else:self.override=None;self.targets[2]=self.tip.copy();self.rotations[2]=self.toolR;self.switch('tool_seat')
    else:
     if self.route_index>0:self.route_index-=1
     else:self.override=None;self.targets[2]=tcp[2].copy();self.rotations[2]=robot['rotation'][2];self.switch('done')
  elif self.state=='tool_seat':
   from scipy.spatial.transform import Rotation
   good=np.linalg.norm(tcp[2]-self.tip)<.004 and np.linalg.norm(Rotation.from_matrix(self.toolR@robot['rotation'][2].T).as_rotvec())<.035
   self.settled=self.settled+.02 if good else 0.
   if self.settled>.5:
    self.events.append({'member':self.member,'time_s':robot['time'],'kind':'abstract_panel_to_frame_nail','physical_bond':False,'tip_m':tcp[2].tolist()});self.targets[2]=self.tip+[0,0,.08];self.switch('tool_retract')
  elif self.state=='tool_retract':
   if np.linalg.norm(tcp[2]-self.targets[2])<.005:self.route_index=len(self.route)-1;self.switch('tool_return')
  if self.time>35:raise RuntimeError('phase_timeout:'+self.state)
  return self.targets.copy(),self.rotations.copy(),self.gripper.copy(),self.override
