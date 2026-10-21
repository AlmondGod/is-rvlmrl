"""Panel-placement station; free sheet and four free members, no attachments.
Starts from the saved assembled-frame poses, not a new full-frame policy trial.
"""
import json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from experiments.frame_assembly_v2.environment import Env as FrameEnv,scene as frame_scene,GOALS,YAWS
from experiments.arm_corner.environment import down_rotation
from experiments.box_assembly.environment import fmt
PANEL_SIZE=np.array([.9744,.9744,.012]);PANEL_START=np.array([.06,0,.286]);FIXTURE_HEIGHT=.12;PANEL_GOAL=np.array([0,0,.070+FIXTURE_HEIGHT])
HOME=np.array([[-.30,-.67,.42],[.30,.67,.42],[0,.45,.70]])
SIDE_R=np.array([np.column_stack(([1,0,0],[0,0,-1],[0,1,0])),np.column_stack(([-1,0,0],[0,0,-1],[0,-1,0]))])
OFFSETS=np.array([[-.30,-PANEL_SIZE[1]/2+.005,0],[.30,PANEL_SIZE[1]/2-.005,0]])
def scene(dt=.001):
 root=ET.fromstring(frame_scene(dt));root.set('model','wall_panel_station');root.find('./visual/headlight').set('ambient','.3 .3 .3');root.find('./visual/headlight').set('diffuse','.3 .3 .3');root.find('./worldbody/light').set('diffuse','.3 .3 .3');w=root.find('worldbody');contact=root.find('contact');act=root.find('actuator')
 ET.SubElement(w,'geom',name='frame_plinth',type='box',pos=fmt([0,0,FIXTURE_HEIGHT/2]),size=fmt([.46,.46,FIXTURE_HEIGHT/2]),rgba='.20 .24 .27 1',friction='.5 .001 .0001',solref='.004 1',solimp='.99 .999 .001 .5 2')
 # Explicit overhead inverted mounting leaves the entire wall surface accessible.
 b=w.find("body[@name='a2_link_base']");b.set('pos','0 0 0');b.set('quat','0 1 0 0');w.remove(b)
 carriage=ET.SubElement(w,'body',name='overhead_carriage',pos='0 0 1.30');ET.SubElement(carriage,'joint',name='overhead_z',type='slide',axis='0 0 1',range='-.55 0',damping='400');carriage.append(b)
 ET.SubElement(act,'position',name='overhead_act',joint='overhead_z',kp='30000',kv='1800',ctrlrange='-.55 0',forcerange='-2000 2000')
 for x in (-1.05,1.05):ET.SubElement(w,'geom',name=f'gantry_post_{x}',type='box',pos=fmt([x,.75,.65]),size='.035 .035 .65',rgba='.22 .24 .25 1')
 ET.SubElement(w,'geom',name='gantry_cross',type='box',pos='0 .75 1.36',size='1.085 .04 .035',rgba='.22 .24 .25 1')
 ET.SubElement(w,'geom',name='gantry_mount',type='box',pos='0 .375 1.36',size='.05 .375 .035',rgba='.22 .24 .25 1')
 for i in range(4):
  b=w.find(f"body[@name='wood{i}']");b.set('pos',fmt(GOALS[i]));b.set('quat',fmt([np.cos(YAWS[i]/2),0,0,np.sin(YAWS[i]/2)]))
 b=ET.SubElement(w,'body',name='panel',pos=fmt(PANEL_START));ET.SubElement(b,'freejoint',name='panel_free');ET.SubElement(b,'geom',name='panel_geom',type='box',size=fmt(PANEL_SIZE/2),density='600',rgba='.72 .62 .45 1',friction='.5 .001 .0001',solref='.004 1',solimp='.99 .999 .001 .5 2')
 # Two visible, actuated staging shelves. They withdraw only after lifting.
 for i,sign in enumerate((-1,1)):
  ET.SubElement(w,'geom',name=f'rack_post{i}',type='box',pos=fmt([.06,sign*.72,.14]),size='.12 .025 .14',rgba='.28 .29 .30 1')
  b=ET.SubElement(w,'body',name=f'rack{i}',pos=fmt([.06,sign*.46,.265]));ET.SubElement(b,'joint',name=f'rack_slide{i}',type='slide',axis=fmt([0,sign,0]),range='0 .20',damping='100')
  # Narrow shelf cantilevers, not invisible support or object attachment.
  ET.SubElement(b,'geom',name=f'rack_shelf{i}',type='box',size='.12 .025 .015',mass='3',rgba='.35 .36 .38 1',friction='.5 .001 .0001',solref='.004 1',solimp='.99 .999 .001 .5 2')
  ET.SubElement(act,'position',name=f'rack_act{i}',joint=f'rack_slide{i}',kp='4000',kv='200',ctrlrange='0 .20',forcerange='-400 400')
  for arm in (0,1):
   for side in ('left','right'):
    for n in (1,2):
     if i==0:ET.SubElement(contact,'pair',geom1='panel_geom',geom2=f'a{arm}_{side}_finger_pad_{n}',condim='3',friction='.65 .65 .001 .0001 .0001',solref='.004 1',solimp='.99 .999 .001 .5 2')
 return ET.tostring(root,encoding='unicode')
class Env(FrameEnv):
 scene_builder=staticmethod(scene)
 def __init__(self,seed=0,dt=.001,offset=(0,0)):
  self.xml=self.scene_builder(dt);self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m);self.ikd=mujoco.MjData(self.m);m,d=self.m,self.d
  self.ids=[np.array([m.joint(f'a{i}_joint{k}').id for k in range(1,8)]) for i in range(3)];self.qids=[m.jnt_qposadr[j] for j in self.ids];self.vids=[m.jnt_dofadr[j] for j in self.ids];self.aids=[np.array([m.actuator(f'a{i}_act{k}').id for k in range(1,8)]) for i in range(3)];self.sites=[m.site(f'a{i}_link_tcp').id for i in range(3)];self.grips=[m.actuator(f'a{i}_gripper').id for i in (0,1)];self.woods=[m.body(f'wood{i}').id for i in range(4)]
  self.panel=m.body('panel').id;self.racks=[m.actuator(f'rack_act{i}').id for i in range(2)];self.rack_command=0.;self.overhead_command=0.;self.overhead_act=m.actuator("overhead_act").id
  saved=json.loads(Path('outputs/frame_assembly/astra_live_v3/result.json').read_text())['metrics']['wood_evaluator_only']
  from scipy.spatial.transform import Rotation
  for i,p in enumerate(saved):
   q=int(m.joint(f'wood{i}_free').qposadr[0]);d.qpos[q:q+3]=np.array(p['center'])+[0,0,FIXTURE_HEIGHT];r=Rotation.from_matrix(p['rotation']).as_quat();d.qpos[q+3:q+7]=r[[3,0,1,2]]
  q=int(m.joint('panel_free').qposadr[0]);d.qpos[q:q+2]+=offset
  self.targets=HOME.copy();self.rotations=np.array([*SIDE_R,down_rotation(0)])
  for i in range(3):d.qpos[self.qids[i]]=[0,-.4,0,1.1,0,.6,0]
  mujoco.mj_forward(m,d);self.ikerrors=[]
  for i in range(3):
   q,err=self.ik(i,self.targets[i],self.rotations[i],iterations=250,multistart=True);d.qpos[self.qids[i]]=q;d.ctrl[self.aids[i]]=q;self.ikerrors.append(err)
  mujoco.mj_forward(m,d);self.qtarget=[d.qpos[q].copy() for q in self.qids];self.integral=np.zeros((3,7));self.gripper=np.zeros(2);self.peak=0.;self.worst={};self.saturation=0;self.contacts=[];self.velocity_limits=np.full(7,.7)
  from experiments.frame_assembly_v2.plan_tool import Planner
  pl=Planner(self)
  # Select collision-free RESET configurations only; no reset path is executed.
  pl.edge=lambda a,b: True
  for arm in range(3):
   pl.arm=arm;pl.qa=self.qids[arm];pl.va=self.vids[arm];pl.ids=self.ids[arm];pl.sid=self.sites[arm];pl.lo=m.jnt_range[pl.ids,0]+.03;pl.hi=m.jnt_range[pl.ids,1]-.03
   for a in range(3):pl.d.qpos[self.qids[a]]=d.qpos[self.qids[a]]
   print("initializing arm",arm,flush=True)
   q,_=pl.solve(self.targets[arm],self.rotations[arm],d.qpos[self.qids[arm]])
   d.qpos[self.qids[arm]]=q;d.ctrl[self.aids[arm]]=q
  mujoco.mj_forward(m,d);self.qtarget=[d.qpos[q].copy() for q in self.qids];self.ik_cache={}
 def step(self,*args,**kwargs):
  self.d.ctrl[self.racks]=self.rack_command
  self.d.ctrl[self.overhead_act]=self.overhead_command+self.d.qfrc_bias[int(self.m.joint("overhead_z").dofadr[0])]/30000
  return super().step(*args,**kwargs)
 def panel_metrics(self):
  m,d=self.m,self.d;forces={'grip':[0.,0.],'rack':0.,'frame':0.,'other':0.};f=np.zeros(6)
  for k,c in enumerate(d.contact):
   names=[m.geom(g).name or '' for g in (c.geom1,c.geom2)]
   if 'panel_geom' not in names:continue
   mujoco.mj_contactForce(m,d,k,f);value=max(0.,float(f[0]))
   if any('finger_pad' in n for n in names):
    for i in (0,1):
     if any(n.startswith(f'a{i}_') for n in names):forces['grip'][i]+=value
   elif any(n.startswith('rack') for n in names):forces['rack']+=value
   elif any(n.startswith('wood') for n in names):forces['frame']+=value
   else:forces['other']+=value
  R=d.xmat[self.panel].reshape(3,3);pos=d.xpos[self.panel];frame=super().metrics()
  frame['support_plane_m']=FIXTURE_HEIGHT;frame['frame_geometry_ok']=bool(all(c['ok'] for c in frame['corners']) and max(frame['position_error_m'])<.006 and max(frame['yaw_error_deg'])<1 and max(frame['tilt_deg'])<1 and max(abs(p['center'][2]-(.032+FIXTURE_HEIGHT)) for p in frame['wood_evaluator_only'])<.003)
  return {'center_evaluator_only':pos.tolist(),'rotation_evaluator_only':R.tolist(),'xy_error_m':float(np.linalg.norm(pos[:2])),'height_error_m':float(pos[2]-PANEL_GOAL[2]),'tilt_deg':float(np.degrees(np.arccos(np.clip(R[2,2],-1,1)))),'forces':forces,'frame':frame,'panel_mass_kg':float(m.body_mass[self.panel])}
