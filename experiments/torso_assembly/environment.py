"""Continuous free-body assembly; native UR10e arms on a fixed torso pillar.
Provisional adapter/mount/fixture; no workpiece attachment or post-reset pose writes.
"""
import copy,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from experiments.wall_panel.environment import scene as panel_scene,Env as PanelEnv,PANEL_SIZE,PANEL_START,PANEL_GOAL,FIXTURE_HEIGHT
from experiments.frame_assembly_v2.environment import Env as FrameEnv,GOALS as G,STARTS as S,YAWS,SIZES,OUTER,W,H,GAP,CORNER_PAIRS
from experiments.arm_corner.environment import down_rotation
from experiments.box_assembly.environment import fmt
GOALS=G+[0,0,FIXTURE_HEIGHT]
# Two separated feeder lanes on each side; the central frame starts empty.
STARTS=np.array([[-.84,0,H/2+FIXTURE_HEIGHT],[-.68,0,H/2+FIXTURE_HEIGHT],[.84,0,H/2+FIXTURE_HEIGHT],[.68,0,H/2+FIXTURE_HEIGHT]])
START_YAWS=np.zeros(4);ASSIGN=[0,0,1,1]
SHUTTLE_START=1.55
HOME=np.array([[-.9,.65,.65],[.9,.65,.65],[0,-.5,1.7]])
SIDE_R=np.array([np.column_stack(([0,1,0],[0,0,1],[1,0,0])),np.column_stack(([0,-1,0],[0,0,1],[-1,0,0]))])
OFFSETS=np.array([[-PANEL_SIZE[0]/2+.005,0,0],[PANEL_SIZE[0]/2-.005,0,0]])
UR=Path(__file__).resolve().parents[2]/'third_party/mujoco_menagerie/universal_robots_ur10e'

def scene(dt=.001,layout_seed=None,fixture_half_width=.46,shoulder_half_span=.34):
 root=ET.fromstring(panel_scene(dt));root.set('model','torso_continuous_frame_wall');w=root.find('worldbody');act=root.find('actuator');con=root.find('contact')
 # Remove portal and its moving carriage; all robot mounts belong to one pillar.
 carriage=w.find("body[@name='overhead_carriage']");old2=carriage.find("body[@name='a2_link_base']");carriage.remove(old2);w.remove(carriage)
 act.remove(act.find("position[@name='overhead_act']"))
 for g in list(w):
  if g.get('name','').startswith('gantry_'):w.remove(g)
 torso=ET.SubElement(w,'body',name='robot_torso')
 def box(parent,name,pos,size,color='.17 .26 .32 1'):
  ET.SubElement(parent,'geom',name=name,type='box',pos=fmt(pos),size=fmt(size),rgba=color)
 box(torso,'torso_foot',[0,.92,-.48],[.48,.42,.06]);box(torso,'torso_pillar',[0,.92,.565],[.16,.14,.985])
 box(torso,'torso_shoulders',[0,.85,.36],[shoulder_half_span+.08,.12,.035]);
 for x in (-shoulder_half_span,shoulder_half_span):box(torso,'torso_shoulder_bracket_'+str(x),[x,.66,.36],[.10,.19,.035])
 box(torso,'torso_upper_mount',[0,.585,1.59],[.18,.515,.035])
 for a in range(3):
  old=old2 if a==2 else w.find(f"body[@name='a{a}_link_base']")
  end=copy.deepcopy(old.find(f".//body[@name='a{a}_xarm_gripper_base_link']") if a<2 else old.find(".//body[@name='a2_nailer']"))
  if a<2:w.remove(old)
  for k in range(1,8):act.remove(act.find(f"general[@name='a{a}_act{k}']"))
  r=ET.parse(UR/'ur10e.xml').getroot();prefix=f'a{a}_ur_'
  for mesh in r.findall('./asset/mesh'):
   mesh.set('name',Path(mesh.get('file')).stem);mesh.set('file',str(UR/'assets'/mesh.get('file')))
  for el in r.iter():
   for key in ('name','class','childclass','mesh','material','joint','body1','body2'):
    if key in el.attrib:el.set(key,prefix+el.get(key))
  base=r.find('./worldbody/body');base.set('name',f'a{a}_link_base');base.set('pos',fmt([(-shoulder_half_span if a==0 else shoulder_half_span if a==1 else 0),.55 if a<2 else .15,.395 if a<2 else 1.55]));base.set('quat','1 0 0 0' if a<2 else '0 1 0 0')
  for k,j in enumerate(base.iter('joint'),1):
   oldname=j.get('name');j.set('name',f'a{a}_joint{k}')
   ac=next(x for x in r.find('actuator') if x.get('joint')==oldname);ac.set('joint',j.get('name'));ac.set('name',f'a{a}_act{k}')
  for k,g in enumerate(base.iter('geom')):g.set('name',f'a{a}_arm_geom{k}')
  wrist=base.find(f".//body[@name='{prefix}wrist_3_link']")
  adapter=ET.SubElement(wrist,'body',name=f'a{a}_tool_adapter',pos='0 .12 0',quat='-.70710678 .70710678 0 0')
  ET.SubElement(adapter,'geom',name=f'a{a}_adapter_geom',type='cylinder',pos='0 0 -.01',size='.038 .01',mass='.15',rgba='.25 .25 .25 1')
  end.set('pos','0 0 0');end.set('quat','1 0 0 0')
  if a<2:end.set('childclass',f'a{a}_xarm7')
  adapter.append(end);torso.append(base)
  for tag in ('asset','actuator'):
   for el in r.find(tag):root.find(tag).append(el)
  for el in r.find('default'):root.find('default').append(el)
 # All four members are loose from the same initial scene, supported at z=.12.
 for i in range(4):
  body=w.find(f"body[@name='wood{i}']");body.set('pos',fmt(STARTS[i]));body.set('quat',fmt([np.cos(START_YAWS[i]/2),0,0,np.sin(START_YAWS[i]/2)]))
 w.find("geom[@name='frame_plinth']").set('size',fmt([fixture_half_width,.46,.06]))
 if fixture_half_width<.46:
  # Two side access slots, while end ledges still support every member.
  for side in (-1,1):
   for end in (-1,1):
    name=f'fixture_ledge_{side}_{end}';box(w,name,[side*(fixture_half_width+.46)/2,end*.28,.06],[(.46-fixture_half_width)/2,.18,.06],'.20 .24 .27 1')
    # Same physical support surface/contact compliance as the original plinth.
    for key in ('friction','solref','solimp'):w.find(f"geom[@name='{name}']").set(key,w.find("geom[@name='frame_plinth']").get(key))
 # Separate feeder supports leave the central sheet-grasp corridors clear.
 for i in range(4):
  u=np.array([-np.sin(START_YAWS[i]),np.cos(START_YAWS[i]),0.])
  for side in (-1,1):
   pos=STARTS[i]+u*(side*.30);pos[2]=.06
   box(w,f'feeder_support_{i}_{side}',pos,[.055,.055,.06],'.20 .24 .27 1')
 for side in (-1,1):
  box(w,f'feeder_deck_{side}',[side*.76,0,-.03],[.20,.51,.03],'.20 .24 .27 1')
  for end in (-1,1):box(w,f'feeder_leg_{side}_{end}',[side*.72,end*.38,-.30],[.035,.035,.24],'.20 .24 .27 1')
 w.find("geom[@name='bench']").set('size','.64 .64 .045');w.find("geom[@name='bench']").set('pos','0 0 -.045')
 # Sheet starts off to the right. A visible motorized fixture feeds it AFTER
 # frame fastening; friction carries the free sheet. No state insertion/reset.
 panel=w.find("body[@name='panel']");panel.set('pos',fmt(PANEL_START+[SHUTTLE_START,0,0]))
 shuttle=ET.SubElement(w,'body',name='panel_shuttle',pos='0 0 0');ET.SubElement(shuttle,'joint',name='shuttle_x',type='slide',axis='1 0 0',range=f'0 {SHUTTLE_START}',damping='150');ET.SubElement(shuttle,'inertial',pos='0 0 -.3',mass='15',diaginertia='1 1 1')
 for el in list(w):
  if el.get('name','').startswith('rack'):
   w.remove(el);shuttle.append(el)
 box(shuttle,'shuttle_cross',[.85,0,-.36],[.06,.75,.035],'.3 .32 .35 1')
 for y in (-.72,.72):box(shuttle,'shuttle_fork_'+str(y),[.455,y,-.36],[.395,.035,.035],'.3 .32 .35 1')
 for i in range(2):
  post=shuttle.find(f"geom[@name='rack_post{i}']");xyz=np.fromstring(post.get('pos'),sep=' ');xyz[2]=-.04;post.set('pos',fmt(xyz));post.set('size','.12 .025 .32')
 for y in (-.72,.72):box(w,'feed_rail_'+str(y),[.735,y,-.45],[.9,.04,.04],'.35 .37 .39 1')
 ET.SubElement(act,'position',name='shuttle_act',joint='shuttle_x',kp='8000',kv='900',ctrlrange=f'0 {SHUTTLE_START}',forcerange='-1000 1000')
 ET.SubElement(w,'geom',name='floor',type='plane',pos='0 0 -.54',size='4 4 .1',rgba='.12 .13 .14 1')
 for i,(x,y) in enumerate(((-.5,-.5),(.5,-.5),(-.5,.5),(.5,.5))):box(w,f'table_leg{i}',[x,y,-.3],[.035,.035,.24],'.2 .22 .24 1')
 pos=np.array([.25,-2.4,2.3]);look=np.array([.25,0,.15]);back=pos-look;back/=np.linalg.norm(back);right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
 ET.SubElement(w,'camera',name='torso_overview',pos=fmt(pos),xyaxes=fmt(np.r_[right,up]),fovy='58')
 pos=np.array([3.2,-4.3,3.]);look=np.array([.35,.1,.9]);back=pos-look;back/=np.linalg.norm(back);right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
 ET.SubElement(w,'camera',name='torso_context',pos=fmt(pos),xyaxes=fmt(np.r_[right,up]),fovy='48')
 pos=np.array([0,-.9,2.4]);back=pos-[0,0,.15];back/=np.linalg.norm(back);right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
 ET.SubElement(w,'camera',name='camera3',pos=fmt(pos),xyaxes=fmt(np.r_[right,up]),fovy='45')
 pos=np.array([0,-1.2,1.]);back=pos-[0,0,.15];back/=np.linalg.norm(back);right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
 low=w.find("camera[@name='camera2']");low.set('pos',fmt(pos));low.set('xyaxes',fmt(np.r_[right,up]));low.set('fovy','55')
 # Camera below the torso beam, offset from the nailer's sampled path.
 # Housing/bracket feasibility still needs continuous physical validation.
 top=w.find("camera[@name='camera0']");top.set('pos','.35 -.2 1.25');top.set('fovy','85')
 # Evaluator-owned initial condition; never passed to perception/planning.
 if layout_seed is not None:
  rng=np.random.default_rng(layout_seed)
  for side in ((0,1),(2,3)):
   lanes=rng.permutation(side)
   for i,lane in zip(side,lanes):
    p=STARTS[lane].copy();p[:2]+=rng.uniform([-.012,-.035],[.012,.035]);yaw=rng.uniform(-.06,.06)
    body=w.find(f"body[@name='wood{i}']");body.set('pos',fmt(p));body.set('quat',fmt([np.cos(yaw/2),0,0,np.sin(yaw/2)]))
 return ET.tostring(root,encoding='unicode')

class Env(PanelEnv):
 def __init__(self,dt=.001,layout_seed=None,physical_nails=False,fixture_half_width=.46,shoulder_half_span=.34):
  self.fixture_half_width=fixture_half_width;self.shoulder_half_span=shoulder_half_span;self.xml=scene(dt,layout_seed,fixture_half_width,shoulder_half_span);self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m);self.ikd=mujoco.MjData(self.m);m,d=self.m,self.d
  self.ids=[np.array([m.joint(f'a{i}_joint{k}').id for k in range(1,7)]) for i in range(3)];self.qids=[m.jnt_qposadr[j] for j in self.ids];self.vids=[m.jnt_dofadr[j] for j in self.ids];self.aids=[np.array([m.actuator(f'a{i}_act{k}').id for k in range(1,7)]) for i in range(3)];self.sites=[m.site(f'a{i}_link_tcp').id for i in range(3)];self.grips=[m.actuator(f'a{i}_gripper').id for i in (0,1)];self.woods=[m.body(f'wood{i}').id for i in range(4)];self.panel=m.body('panel').id
  self.racks=[m.actuator(f'rack_act{i}').id for i in range(2)];self.rack_command=0.;self.shuttle_command=SHUTTLE_START;d.qpos[int(m.joint('shuttle_x').qposadr[0])]=SHUTTLE_START
  self.targets=HOME.copy();self.rotations=np.array([down_rotation(np.pi/2),down_rotation(np.pi/2),down_rotation(0)])
  for a in range(3):d.qpos[self.qids[a]]=[-1.57,-1.57,1.57,-1.57,-1.57,0]
  self.qtarget=[d.qpos[q].copy() for q in self.qids];self.integral=np.zeros((3,6));self.gripper=np.zeros(2);self.peak=0.;self.worst={};self.saturation=0;self.contacts=[];self.velocity_limits=np.full(6,.7);self.ik_cache={}
  mujoco.mj_forward(m,d)
  from .planner import Planner
  pl=Planner(self);pl.edge=lambda a,b:True
  for arm in range(3):
   pl.select(arm);print('initialize',arm,flush=True);q,_=pl.solve(self.targets[arm],self.rotations[arm],d.qpos[self.qids[arm]]);d.qpos[self.qids[arm]]=q;d.ctrl[self.aids[arm]]=q;pl.d.qpos[self.qids[arm]]=d.qpos[self.qids[arm]]
  mujoco.mj_forward(m,d);self.qtarget=[d.qpos[q].copy() for q in self.qids]
  self.nails=None
  if physical_nails:
   from .fasteners import NailBank
   self.nails=NailBank(self);self._joint_internal_forces=self.nails.apply
 def ik(self,arm,p,R,iterations=40,multistart=False):
  if getattr(self,'joint_override',None) is not None and arm in self.joint_override:
   self.ik_cache.pop(arm,None);delta=self.joint_override[arm]-self.qtarget[arm];return self.qtarget[arm]+delta*min(1.,.5*.02/max(np.max(abs(delta)),1e-9)),0.
  cached=self.ik_cache.get(arm)
  if cached is not None and np.max(abs(cached[0]-p))<1e-10 and np.max(abs(cached[1]-R))<1e-10:return cached[2].copy(),cached[3]
  d=self.ikd;m=self.m
  for a in range(3):d.qpos[self.qids[a]]=self.d.qpos[self.qids[a]]
  qa=self.qids[arm];va=self.vids[arm];sid=self.sites[arm];q=self.qtarget[arm].copy();jp=np.zeros((3,m.nv));jr=jp.copy()
  for _ in range(iterations):
   d.qpos[qa]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);err=np.r_[p-d.site_xpos[sid],.3*Rotation.from_matrix(R@d.site_xmat[sid].reshape(3,3).T).as_rotvec()]
   if np.linalg.norm(err)<1e-6:break
   mujoco.mj_jacSite(m,d,jp,jr,sid);J=np.vstack((jp[:,va],.3*jr[:,va]));dq=J.T@np.linalg.solve(J@J.T+.0001*np.eye(6),err);q=np.clip(q+np.clip(dq,-.08,.08),m.jnt_range[self.ids[arm],0]+.03,m.jnt_range[self.ids[arm],1]-.03)
  self.ik_cache[arm]=(np.array(p).copy(),np.array(R).copy(),q.copy(),float(np.linalg.norm(err)));return q,float(np.linalg.norm(err))
 def step(self,*args,**kwargs):
  self.d.ctrl[self.racks]=self.rack_command;self.d.ctrl[self.m.actuator('shuttle_act').id]=self.shuttle_command
  return FrameEnv.step(self,*args,**kwargs)
 def panel_metrics(self):return PanelEnv.panel_metrics(self)

 def images(self,renderer):
  # Only the three cameras consumed by this controller are rendered. Empty
  # slots are explicit; no duplicated images are presented as extra sensors.
  frames=[None]*4;cameras=[None]*4;h,w=renderer.height,renderer.width
  for i in (0,2,3):
   renderer.update_scene(self.d,camera=f'camera{i}');frames[i]=renderer.render().copy();c=self.m.camera(f'camera{i}').id;R=np.diag([1.,-1.,-1.])@self.d.cam_xmat[c].reshape(3,3).T;T=np.eye(4);T[:3,:3]=R;T[:3,3]=-R@self.d.cam_xpos[c];f=.5*h/np.tan(np.deg2rad(self.m.cam_fovy[c])/2);cameras[i]={'K':[[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]],'world_to_camera':T.tolist()}
  return frames,cameras
