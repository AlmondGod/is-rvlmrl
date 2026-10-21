import numpy as np
import mujoco
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from .environment import SHUTTLE_START,scene
from experiments.frame_assembly_v2.plan_tool import Planner as Base
class Planner(Base):
 def __init__(self,e):
  # Own planning world built from robot/fixture CAD, not the episode model.
  # All workpiece reset poses are erased before planning is possible.
  m=mujoco.MjModel.from_xml_string(scene(fixture_half_width=getattr(e,"fixture_half_width",.46),shoulder_half_span=getattr(e,"shoulder_half_span",.34)));d=mujoco.MjData(m)
  for a in range(3):d.qpos[e.qids[a]]=e.robot()['joints'][a]
  proxy=SimpleNamespace(m=m,d=d,qids=e.qids,vids=e.vids,ids=e.ids,sites=e.sites,robot=e.robot)
  super().__init__(proxy)
  for i in range(4):
   q=int(m.joint(f'wood{i}_free').qposadr[0]);self.d.qpos[q:q+3]=[0,0,-10-i];self.d.qpos[q+3:q+7]=[1,0,0,0]
  self.panel_nominal_size=m.geom('panel_geom').size.copy();self.panel_seen=False
  self.reserve_unobserved_sheet(SHUTTLE_START)
  self.d.qpos[int(self.m.joint('shuttle_x').qposadr[0])]=SHUTTLE_START
  for a in range(3):self.d.qpos[self.qids[a]]=e.robot()['joints'][a]
 def reserve_unobserved_sheet(self,shuttle_x):
  # Conservative surveyed feeder occupancy envelope, NOT a sheet pose estimate.
  # It travels with fixture encoders; fresh RGB replaces it before sheet grasp.
  q=int(self.m.joint('panel_free').qposadr[0]);self.d.qpos[q:q+3]=[shuttle_x+.06,0,.285];self.d.qpos[q+3:q+7]=[1,0,0,0]
  self.m.geom('panel_geom').size[:]=[.54,.55,.065]
 def select(self,a):
  self.arm=a;self.qa=self.qids[a];self.va=self.vids[a];self.ids=self.allids[a];self.sid=self.sites[a];self.lo=self.m.jnt_range[self.ids,0]+.03;self.hi=self.m.jnt_range[self.ids,1]-.03
 def sync(self,robot,est,panel=None):
  if len(est)!=4 or any(e is None for e in est):raise ValueError('Planning requires four visually observed member poses; no reset fallback')
  if panel is None and not self.panel_seen:self.reserve_unobserved_sheet(robot.get('shuttle_x',SHUTTLE_START))
  for a in range(3):self.d.qpos[self.qids[a]]=robot['joints'][a]
  for name in ('shuttle_x','rack_slide0','rack_slide1'):
   if name in robot:self.d.qpos[int(self.m.joint(name).qposadr[0])]=robot[name]
  for i,e in enumerate(est):
   if e is None:continue
   q=int(self.m.joint(f'wood{i}_free').qposadr[0]);self.d.qpos[q:q+3]=e['center'];self.d.qpos[q+3:q+7]=[np.cos(e['yaw']/2),0,0,np.sin(e['yaw']/2)]
  if panel is not None:
   self.panel_seen=True;self.m.geom('panel_geom').size[:]=self.panel_nominal_size
   q=int(self.m.joint('panel_free').qposadr[0]);self.d.qpos[q:q+3]=panel['center'];rot=Rotation.from_matrix(panel['rotation']).as_quat();self.d.qpos[q+3:q+7]=rot[[3,0,1,2]]
 def plan(self,arm,robot,est,p,R,panel=None):
  self.select(arm);self.sync(robot,est,panel);return self.solve(p,R,np.array(robot['joints'][arm]))[1]
 def plan_flip_approach(self,arm,robot,est,p,R,panel=None):
  self.select(arm);self.sync(robot,est,panel)
  def outside(q):
   # Robot CAD/encoders only. Keep the forearm on the outside of the
   # assembly's rotation disk; the usual nearest IK branch puts it inside.
   self.d.qpos[self.qa]=q;mujoco.mj_forward(self.m,self.d)
   return (-1 if arm==0 else 1)*np.sin(q[0])>0 and (-1 if arm==0 else 1)*self.d.xpos[self.m.body(f'a{arm}_ur_forearm_link').id,0]>.60
  return self.solve(p,R,np.array(robot['joints'][arm]),goal_filter=outside)[1]
 def plan_home(self,arm,robot,est):
  self.sync(robot,est);return super().plan_home(arm,robot,est)
 def plan_carry(self,arm,piece,robot,est,center,R):
  self.select(arm);self.sync(robot,est)
  initial_R=np.array(robot['rotation'][arm]);offset=initial_R.T@(est[piece]['center']-robot['tcp'][arm]);yaw=est[piece]['yaw'];wood_R=Rotation.from_euler('z',yaw).as_matrix()
  self.carry=(piece,offset,initial_R.T@wood_R)
  try:return self.solve(np.array(center)-R@offset,R,np.array(robot['joints'][arm]))[1]
  finally:self.carry=None
 def plan_pick(self,arm,robot,est,p,R):
  self.descent_check=(np.array(p).copy(),np.array(R).copy())
  try:return self.plan(arm,robot,est,p,R)
  finally:self.descent_check=None
 def valid(self,q):
  import mujoco
  carry=getattr(self,'carry',None)
  if carry is not None:
   # A rigid no-slip grasp is a planning approximation only. The actual
   # scene still carries the free body through finite-force finger contacts.
   self.d.qpos[self.qa]=q;mujoco.mj_forward(self.m,self.d);piece,offset,relative=carry;R=self.d.site_xmat[self.sid].reshape(3,3);j=int(self.m.joint(f'wood{piece}_free').qposadr[0]);self.d.qpos[j:j+3]=self.d.site_xpos[self.sid]+R@offset;quat=Rotation.from_matrix(R@relative).as_quat();self.d.qpos[j+3:j+7]=quat[[3,0,1,2]];mujoco.mj_forward(self.m,self.d)
   wood=f'wood{piece}_geom'
   for c in self.d.contact:
    names=[self.m.geom(g).name or '' for g in (c.geom1,c.geom2)]
    if c.dist>=-.000005:continue
    if wood in names and any(n.startswith(f'a{self.arm}_') and 'finger_pad' in n for n in names):continue
    if wood in names or any(n.startswith(f'a{self.arm}_') for n in names):return False
   return True
  if not super().valid(q):return False
  check=getattr(self,'descent_check',None)
  if check is None or np.linalg.norm(self.d.site_xpos[self.sid]-check[0])>.0002:return True
  # A reachable, collision-free approach pose is insufficient: check the
  # entire vertical grasp descent on the same IK branch before accepting it.
  p,R=check;jp=np.zeros((3,self.m.nv));jr=jp.copy();cur=q.copy()
  for dz in np.linspace(0,.214,35)[1:]:
   goal=p-[0,0,dz]
   for _ in range(35):
    self.d.qpos[self.qa]=cur;mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d);err=np.r_[goal-self.d.site_xpos[self.sid],.3*Rotation.from_matrix(R@self.d.site_xmat[self.sid].reshape(3,3).T).as_rotvec()]
    if np.linalg.norm(err)<.00002:break
    mujoco.mj_jacSite(self.m,self.d,jp,jr,self.sid);J=np.vstack((jp[:,self.va],.3*jr[:,self.va]));cur=np.clip(cur+np.clip(J.T@np.linalg.solve(J@J.T+.0001*np.eye(6),err),-.08,.08),self.lo,self.hi)
   if np.linalg.norm(err)>.0001 or not super().valid(cur):
    if getattr(self,'debug_count',0)<5:
     print('descent rejected',self.arm,dz,float(np.linalg.norm(err)),[(self.m.geom(c.geom1).name,self.m.geom(c.geom2).name,float(c.dist)) for c in self.d.contact if c.dist<-.000005 and any((self.m.geom(g).name or '').startswith(f'a{self.arm}_') for g in (c.geom1,c.geom2))],flush=True);self.debug_count=getattr(self,'debug_count',0)+1
    self.d.qpos[self.qa]=q;return False
  self.d.qpos[self.qa]=q;return True
