"""Bounded image-feedback placement and abstract nailgun operation.
No evaluator state. Blueprint positions are desired poses, not measurements.
"""
import numpy as np
import json,time
from pathlib import Path
from scipy.spatial.transform import Rotation
from .environment import GOALS,YAWS,ASSIGN,HOME,OUTER,W,H,GAP,CORNER_PAIRS,down_rotation,FIXTURE_HEIGHT

def nail_pose(corner):
 left=corner in (0,3); south=corner in (0,1);sign=-1 if south else 1
 p=np.array([(-1 if left else 1)*(OUTER-W)/2,sign*(OUTER/2-W-GAP-.018),H])
 direction=np.array([0,sign*.8,-.6]);x=np.array([1.,0,0]);y=np.cross(direction,x);R=np.column_stack((x,y,direction))
 return p,R

def visual_frame_ok(est):
 from experiments.frame_assembly_v2.geometry import visual_corners
 return all(e is not None and np.linalg.norm(e['center'][:2]-GOALS[i,:2])<.006 and abs((e['yaw']-YAWS[i]+np.pi/2)%np.pi-np.pi/2)<np.deg2rad(1) and abs(e['center'][2]-(H/2+FIXTURE_HEIGHT))<.005 for i,e in enumerate(est)) and all(c['ok'] for c in visual_corners(est))

def placement_rotation(estimate,robot,arm,goal_yaw):
 """Correct the visually estimated grasp offset, not just wrist yaw."""
 R=np.array(estimate.get('rotation',Rotation.from_euler('z',estimate['yaw']).as_matrix()))
 grip=np.array(estimate.get('tcp_rotation_at_measurement',robot['rotation'][arm]))
 yaw=goal_yaw
 return Rotation.from_euler('z',yaw).as_matrix()@R.T@grip

class Skills:
 def __init__(self,robot,planner=None):
  self.planner=planner;self.targets=robot['tcp'].copy();self.rotations=robot['rotation'].copy();self.gripper=np.zeros(2);self.state='idle';self.time=0.;self.failure=None;self.holding=[False]*4;self.arm=0;self.piece=0;self.action=None;self.events=[];self.settled=0.;self.joint_override=None;self.planning_times=[]
 def start(self,action,est,robot):
  if self.state not in ('idle','done','failed'):raise ValueError('Busy')
  kind=action['skill'];self.time=0.;self.failure=None;self.action=action;self.targets=robot['tcp'].copy();self.rotations=robot['rotation'].copy();self.settled=0.;self.visual_integral=np.zeros(2)
  if kind in ('pick','position','place','release'):
   self.piece=int(action['piece']);self.arm=ASSIGN[self.piece];p=self.piece;a=self.arm;self.lift_height=FIXTURE_HEIGHT+(.115 if p==2 else .13)
   # A rectangular member and symmetric gripper admit two equivalent yaw
   # directions. The right arm turns outward rather than folding its wrist
   # into its upper arm when transporting the north rail from side staging.
   self.place_yaw=YAWS[p]-(np.pi if p==2 else 0.)
   if est[p] is None:raise ValueError('No fresh visual estimate')
   if kind in ('pick','position'):
    if any(self.holding[i] for i in range(4) if ASSIGN[i]==a):raise ValueError('Arm already carries a member')
    self.state='approach';self.start_z=est[p]['center'][2];self.rotations[a]=down_rotation(est[p]['yaw']);self.targets[a]=est[p]['center']+[0,0,.20]
    self.approach_target=self.targets[a].copy();self.approach_R=self.rotations[a].copy();self.approach_route=np.array(self.planner.plan_pick(a,robot,est,self.targets[a],self.rotations[a]));self.approach_index=1;self.state='approach_route'
   elif kind=='place':
    if not self.holding[p]:raise ValueError('Pick before placement')
    self.state='align';self.rotations[a]=down_rotation(self.place_yaw)
   else:
    if not self.holding[p]:raise ValueError('Nothing held')
    if np.linalg.norm(est[p]['center'][:2]-GOALS[p,:2])>.006 or abs(est[p]['center'][2]-(H/2+FIXTURE_HEIGHT))>.008:raise ValueError('Release rejected: member has not converged to placement')
    self.state='release';self.gripper[a]=0
  elif kind=='nail':
   c=int(action['corner'])
   if not 0<=c<4:raise ValueError('Unknown corner')
   if any(ev['corner']==c for ev in self.events):raise ValueError('Corner already has its one abstract nail')
   if any(self.holding) or not visual_frame_ok(est):raise ValueError('Need released, visually aligned four-member frame')
   self.arm=2;self.corner=c;self.tip,self.toolR=nail_pose(c)
   self.route=np.array(json.loads((Path(__file__).parent/'tool_routes.json').read_text())['routes'][str(c)])
   if np.max(abs(robot['joints'][2]-self.route[0]))>.08:raise ValueError('Nailer is not at verified route home')
   self.route_index=1;self.joint_override={2:self.route[1]};self.state='tool_route'
  elif kind=='stop':self.state='done'
  else:raise ValueError('Unknown primitive')
 def command(self,robot,est):
  previous_targets=self.targets.copy()
  self.time+=.02;a=self.arm;p=self.piece;tcp=robot['tcp'];e=est[p];feedback_gain=1. if e is not None and e.get('full_multiview_silhouette') else 3.;kind=self.action['skill'] if self.action else None
  def switch(s):self.state=s;self.time=0.;self.settled=0.
  def fail(msg):self.failure=msg;self.targets[a]=tcp[a].copy();switch('failed')
  if self.state in ('approach','descend','close','probe','lift','align','lower','settle') and (e is None or robot['time']-e['time']>.4):
   if getattr(self,'vision_hold',None) is None:self.vision_hold=np.array(robot['joints'][a]).copy()
   self.joint_override={a:self.vision_hold}
   if self.time>2:fail('lost_visual_estimate')
   return self.targets,self.rotations,self.gripper,self.joint_override
  if getattr(self,'vision_hold',None) is not None:self.vision_hold=None;self.joint_override=None
  # Bounded XY integral removes persistent servo bias without amplifying
  # each image's noise. Only update with a current accepted two-view pose.
  if self.state in ('align','lower','settle') and e is not None and e.get('full_multiview_silhouette'):
   self.visual_integral=np.clip(self.visual_integral+.5*.02*(GOALS[p,:2]-e['center'][:2]),-.003,.003)
  if self.state=='approach_route':
   self.joint_override={a:self.approach_route[self.approach_index]};self.targets[a]=tcp[a].copy();self.rotations[a]=robot['rotation'][a].copy()
   if np.max(abs(robot['joints'][a]-self.approach_route[self.approach_index]))<.008:
    self.approach_index+=1
    if self.approach_index>=len(self.approach_route):self.joint_override=None;self.targets[a]=self.approach_target.copy();self.rotations[a]=self.approach_R.copy();switch('approach')
   elif self.time>40:fail('approach_route_timeout')
  elif self.state=='approach':
   if np.linalg.norm(tcp[a]-self.targets[a])<.005:switch('descend')
   elif self.time>15:fail('approach_unreachable')
  elif self.state=='descend':
   self.targets[a,:2]=e['center'][:2];self.targets[a,2]=max(self.start_z-.014,self.targets[a,2]-.035*.02)
   if abs(tcp[a,2]-(self.start_z-.014))<.003:switch('close')
   elif self.time>10:fail('descent_tracking')
  elif self.state=='close':
   self.gripper[a]=180
   if self.time>2:self.probe_start=e['center'].copy();self.tcp_start=tcp[a].copy();self.tcp_R_start=robot['rotation'][a].copy();switch('probe')
  elif self.state=='probe':
   self.targets[a,2]=min(self.tcp_start[2]+.04,self.targets[a,2]+.025*.02)
   if e['center'][2]-self.probe_start[2]>.02:self.holding[p]=True;switch('lift')
   elif self.time>4:fail('probe_no_visual_lift')
  elif self.state=='lift':
   self.targets[a,2]+=np.clip(self.lift_height-e['center'][2],-.03*.02,.03*.02)
   if abs(e['center'][2]-self.lift_height)<.004 and self.time>1:
    if self.action['skill']=='position':
     self.carry_R=placement_rotation(e,robot,a,self.place_yaw) if 'rotation' in e else down_rotation(self.place_yaw);goal=GOALS[p].copy();goal[2]=self.lift_height;self.carry_route=np.array(self.planner.plan_carry(a,p,robot,est,goal,self.carry_R));self.carry_index=1;switch('carry_route')
    else:switch('done')
   elif self.time>7:fail('lift_timeout')
  elif self.state=='carry_route':
   self.joint_override={a:self.carry_route[self.carry_index]};self.targets[a]=tcp[a].copy();self.rotations[a]=robot['rotation'][a].copy()
   if np.max(abs(robot['joints'][a]-self.carry_route[self.carry_index]))<.008:
    self.carry_index+=1
    if self.carry_index>=len(self.carry_route):self.joint_override=None;self.rotations[a]=self.carry_R.copy();switch('align')
   elif self.time>70:fail('carry_route_timeout')
  elif self.state=='align':
   desired=GOALS[p].copy();desired[2]=self.lift_height;self.targets[a]=np.array(e.get('tcp_at_measurement',tcp[a]))+feedback_gain*(desired-e['center']);self.rotations[a]=placement_rotation(e,robot,a,self.place_yaw) if 'rotation' in e else down_rotation(self.place_yaw);yawerr=abs((e['yaw']-YAWS[p]+np.pi/2)%np.pi-np.pi/2)
   if np.linalg.norm(e['center'][:2]-GOALS[p,:2])<.0007 and yawerr<.008:switch('lower')
   elif self.time>25:fail('alignment_timeout')
  elif self.state=='lower':
   self.targets[a,:2]=np.array(e.get('tcp_at_measurement',tcp[a]))[:2]+feedback_gain*(GOALS[p,:2]-e['center'][:2]);self.targets[a,2]+=np.clip(H/2+FIXTURE_HEIGHT+.0007-e['center'][2],-.02*.02,.02*.02)
   if abs(e['center'][2]-(H/2+FIXTURE_HEIGHT+.0007))<.001 or (self.targets[a,2]<=self.tcp_start[2]+.0011 and abs(tcp[a,2]-self.tcp_start[2])<.006 and abs(e['center'][2]-(H/2+FIXTURE_HEIGHT))<.008):switch('settle')
   elif self.time>9:fail('lower_timeout')
  elif self.state=='settle':
   # Unlike the old corner test, height alone cannot terminate placement.
   self.targets[a,:2]=np.array(e.get('tcp_at_measurement',tcp[a]))[:2]+feedback_gain*(GOALS[p,:2]-e['center'][:2])
   # Overhead RGB samples about 2.4 mm/pixel at the support surface.
   # A 1.2 mm stop gate demanded subpixel stability against contact friction.
   # Use a 3 mm controller stop gate; the independent 6 mm task criterion
   # and all numerical/contact limits remain unchanged.
   good=np.linalg.norm(e['center'][:2]-GOALS[p,:2])<.003 and abs((e['yaw']-YAWS[p]+np.pi/2)%np.pi-np.pi/2)<.008 and abs(e['center'][2]-(H/2+FIXTURE_HEIGHT))<.006
   self.settled=self.settled+.02 if good else 0.
   if self.settled>1.:switch('done')
   elif self.time>8:fail('final_visual_alignment')
  elif self.state=='release':
   if self.time>2:self.holding[p]=False;self.targets[a,2]+=.20;switch('retreat')
  elif self.state=='retreat':
   if self.time>1 and np.linalg.norm(tcp[a]-self.targets[a])<.008:
    # Side staging changes the grasp orientation. Park through a checked
    # joint route instead of returning to a feeder with the placement yaw.
    started=time.perf_counter();self.parking_route=self.planner.plan_home(a,robot,est);self.planning_times.append(time.perf_counter()-started);self.parking_index=1;switch('park_joint')
   elif self.time>12:fail('post_release_retreat_unreachable')
  elif self.state=='return_feeder':
   if np.linalg.norm(tcp[a]-self.targets[a])<.008:
    if self.planner is None:fail('No collision-aware parking planner')
    else:
     started=time.perf_counter();self.parking_route=self.planner.plan_home(a,robot,est);self.planning_times.append(time.perf_counter()-started);self.parking_index=1;switch('park_joint')
   elif self.time>12:fail('feeder_return_unreachable')
  elif self.state=='park_joint':
   self.targets[a]=tcp[a].copy();self.rotations[a]=robot['rotation'][a].copy();self.joint_override={a:self.parking_route[self.parking_index]}
   if np.max(abs(robot['joints'][a]-self.parking_route[self.parking_index]))<.008:
    self.parking_index+=1
    if self.parking_index>=len(self.parking_route):self.joint_override=None;switch('done')
   elif self.time>25:fail('parking_joint_route_tracking')
  elif self.state=='park':
   if np.linalg.norm(tcp[a]-self.targets[a])<.008:switch('done')
   elif self.time>12:fail('park_unreachable')
  elif self.state in ('tool_route','tool_return'):
   self.targets[2]=tcp[2].copy();self.rotations[2]=robot['rotation'][2].copy();self.joint_override={2:self.route[self.route_index]}
   if np.max(abs(robot['joints'][2]-self.route[self.route_index]))<.008:
    if self.state=='tool_route':
     self.route_index+=1
     if self.route_index>=len(self.route):
      self.joint_override=None;self.targets[2]=self.tip-self.toolR[:,2]*.004;self.rotations[2]=self.toolR;switch('tool_seat')
    else:
     self.route_index-=1
     if self.route_index<0:self.joint_override=None;switch('done')
   elif self.time>25:fail('tool_joint_route_tracking')
  elif self.state=='tool_seat':
   good=visual_frame_ok(est) and np.linalg.norm(tcp[2]-self.tip)<.006 and np.linalg.norm(Rotation.from_matrix(self.toolR@robot['rotation'][2].T).as_rotvec())<.04
   self.settled=self.settled+.02 if good else 0.
   if self.settled>.5:
    self.events.append({'corner':self.corner,'time_s':robot['time'],'tcp_m':tcp[2].tolist(),'tip_target_m':self.tip.tolist(),'kind':'abstract_nail_fired','physical_nail_or_bond':False});self.targets[2]=self.tip-self.toolR[:,2]*.14;switch('tool_retract')
   elif self.time>12:fail('tool_seating_or_frame_alignment')
  elif self.state=='tool_retract':
   if np.linalg.norm(tcp[2]-self.targets[2])<.006:self.route_index=len(self.route)-1;switch('tool_return')
   elif self.time>12:fail('tool_retract_unreachable')
  if self.state in ('align','lower','settle') and e is not None and e.get('full_multiview_silhouette'):
   self.targets[a,:2]+=self.visual_integral
  if self.state in ('lower','settle'):
   # The destination and feeder share a calibrated support plane. Prevent
   # a biased silhouette-height estimate from driving below the grasp's
   # encoder height at that plane. This uses no runtime object state.
   self.targets[a,2]=max(self.targets[a,2],self.tcp_start[2]+.001)
  if kind=='position' and self.state=='done':
   # Completion hands control back to the supervisor, which may need seconds
   # to decide. Hold the last issued, settled target across that handoff.
   # Recomputing it on the terminal tick dropped the feedback correction and
   # could move the still-grasped workpiece while the supervisor was waiting.
   self.targets=previous_targets
  return self.targets.copy(),self.rotations.copy(),self.gripper.copy(),self.joint_override
