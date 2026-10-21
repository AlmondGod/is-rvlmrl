"""Four freely moving frame members, two articulated grippers and one nailer.
Abstract firing records DO NOT attach, snap, weld, or change wood dynamics.
Dimensions and material/actuator parameters are provisional, not calibrated.
"""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from experiments.arm_corner.environment import Env as CornerEnv, scene as corner_scene, down_rotation
from experiments.box_assembly.design import beveled_vertices
from experiments.box_assembly.environment import fmt

OUTER=.9144; W=.038; H=.064; GAP=.002
# Horizontal rails span full outside width; vertical members fit between them.
SIZES=np.array([[W,OUTER,H],[W,OUTER-2*W-2*GAP,H],[W,OUTER,H],[W,OUTER-2*W-2*GAP,H]])
GOALS=np.array([[0,-(OUTER-W)/2,H/2],[-(OUTER-W)/2,0,H/2],[0,(OUTER-W)/2,H/2],[(OUTER-W)/2,0,H/2]])
YAWS=np.array([np.pi/2,0,np.pi/2,0])
STARTS=GOALS.copy(); STARTS[:,:2]*=1.23
BASES=[(-.66,-.66,.02),(.66,.66,.02),(0,0,.02)]
ASSIGN=[0,0,1,1]
HOME=np.array([[-.43,-.43,.34],[.43,.43,.34],[0,.25,.35]])
CORNER_PAIRS=[(0,1),(0,3),(2,3),(2,1)]
CORNER_NAMES=['southwest','southeast','northeast','northwest']

def scene(dt=.001):
 root=ET.fromstring(corner_scene(dt));root.set('model','four_member_frame_abstract_nailing')
 asset=root.find('asset');world=root.find('worldbody');contact=root.find('contact')
 for b in list(world):
  if b.get('name','').startswith('wood'):world.remove(b)
 for mesh in list(asset):
  if mesh.get('name','').startswith('wood'):asset.remove(mesh)
 for pair in list(contact):
  if pair.get('geom1','').startswith('wood'):contact.remove(pair)
 nose=world.find(".//geom[@name='nailer_nose']");nose.set('type','capsule');nose.set('size','.004');nose.set('fromto','0 0 .18 0 0 .306');nose.attrib.pop('pos',None)
 world.find(".//site[@name='a2_link_tcp']").set('pos','0 0 .31')
 for arm,base in enumerate(BASES):world.find(f"body[@name='a{arm}_link_base']").set('pos',fmt(base))
 for i,(size,pos,yaw) in enumerate(zip(SIZES,STARTS,YAWS)):
  name=f'wood{i}';ET.SubElement(asset,'mesh',name=name,vertex=fmt(np.array(beveled_vertices(size,.0005)).ravel()))
  b=ET.SubElement(world,'body',name=name,pos=fmt(pos),quat=fmt([np.cos(yaw/2),0,0,np.sin(yaw/2)]));ET.SubElement(b,'freejoint',name=name+'_free')
  ET.SubElement(b,'geom',name=name+'_geom',mesh=name,type='mesh',density='600',rgba='.72 .49 .28 1',friction='.5 .001 .0001',solref='.004 1',solimp='.99 .999 .001 .5 2')
  for arm in (0,1):
   for side in ('left','right'):
    for n in (1,2):ET.SubElement(contact,'pair',geom1=name+'_geom',geom2=f'a{arm}_{side}_finger_pad_{n}',condim='3',friction='.65 .65 .001 .0001 .0001',solref='.004 1',solimp='.99 .999 .001 .5 2')
 return ET.tostring(root,encoding='unicode')

class Env(CornerEnv):
 def __init__(self,seed=0,dt=.001):
  self.xml=scene(dt);self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m);self.ikd=mujoco.MjData(self.m);m,d=self.m,self.d
  self.ids=[np.array([m.joint(f'a{i}_joint{k}').id for k in range(1,8)]) for i in range(3)];self.qids=[m.jnt_qposadr[j] for j in self.ids];self.vids=[m.jnt_dofadr[j] for j in self.ids];self.aids=[np.array([m.actuator(f'a{i}_act{k}').id for k in range(1,8)]) for i in range(3)];self.sites=[m.site(f'a{i}_link_tcp').id for i in range(3)];self.grips=[m.actuator(f'a{i}_gripper').id for i in (0,1)];self.woods=[m.body(f'wood{i}').id for i in range(4)]
  self.targets=HOME.copy();self.rotations=np.array([down_rotation(np.pi/2),down_rotation(np.pi/2),down_rotation(0)])
  rng=np.random.default_rng(seed)
  for i in range(4):
   qa=int(m.joint(f'wood{i}_free').qposadr[0]);d.qpos[qa:qa+2]+=rng.uniform(-.004,.004,2);yaw=YAWS[i]+rng.uniform(-.015,.015);d.qpos[qa+3:qa+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
  for i in range(3):d.qpos[self.qids[i]]=[0,-.4,0,1.1,0,.6,0]
  mujoco.mj_forward(m,d);self.ikerrors=[]
  for i in range(3):
   q,err=self.ik(i,self.targets[i],self.rotations[i],iterations=250,multistart=True);d.qpos[self.qids[i]]=q;d.ctrl[self.aids[i]]=q;self.ikerrors.append(err)
  mujoco.mj_forward(m,d);self.qtarget=[d.qpos[q].copy() for q in self.qids];self.integral=np.zeros((3,7));self.gripper=np.zeros(2);self.peak=0.;self.worst={};self.saturation=0;self.contacts=[];self.velocity_limits=np.full(7,.7)
 def ik(self,arm,p,R,iterations=40,multistart=False):
  override=getattr(self,'joint_override',None)
  if not hasattr(self,'ik_cache'):self.ik_cache={}
  if override is not None and arm in override and not multistart:
   self.ik_cache.pop(arm,None)
   delta=np.asarray(override[arm])-self.qtarget[arm]
   return self.qtarget[arm]+delta*min(1.,.5*.02/max(np.max(abs(delta)),1e-9)),0.
  cached=self.ik_cache.get(arm)
  if not multistart and cached is not None and np.max(abs(cached[0]-p))<1e-10 and np.max(abs(cached[1]-R))<1e-10:return cached[2].copy(),cached[3]
  q,err=super().ik(arm,p,R,iterations,multistart)
  self.ik_cache[arm]=(np.array(p).copy(),np.array(R).copy(),q.copy(),err)
  return q,err
 def step(self,targets=None,rotations=None,gripper=None,joint_override=None):
  self.joint_override=joint_override
  if joint_override:
   robot=self.robot()
   for arm in joint_override:
    self.targets[arm]=robot['tcp'][arm];self.rotations[arm]=robot['rotation'][arm]
  return super().step(targets,rotations,gripper)
 def metrics(self):
  m,d=self.m,self.d;poses=[];forces=np.zeros(4);support=np.zeros(4);self.contacts=[];f=np.zeros(6)
  for k,c in enumerate(d.contact):
   mujoco.mj_contactForce(m,d,k,f);names=[m.geom(g).name or '' for g in (c.geom1,c.geom2)];self.contacts.append({'names':names,'distance':float(c.dist),'force':f.tolist()})
   for i in range(4):
    if f'wood{i}_geom' in names:
     if any('finger_pad' in n for n in names):forces[i]+=max(0,f[0])
     elif 'bench' in names:support[i]+=max(0,f[0])
  for b in self.woods:poses.append({'center':d.xpos[b].tolist(),'rotation':d.xmat[b].reshape(3,3).tolist()})
  from .geometry import corner_errors
  corners=corner_errors([p['center'] for p in poses],[p['rotation'] for p in poses])
  errors=np.linalg.norm(d.xpos[self.woods,:2]-GOALS[:,:2],axis=1)
  yawerrors=[];tilts=[]
  for i,b in enumerate(self.woods):
   R=d.xmat[b].reshape(3,3);yaw=np.arctan2(-R[0,1],R[1,1]);yawerrors.append(float(np.degrees(abs((yaw-YAWS[i]+np.pi/2)%np.pi-np.pi/2))));tilts.append(float(np.degrees(np.arccos(np.clip(R[2,2],-1,1)))))
  return {'wood_evaluator_only':poses,'position_error_m':errors.tolist(),'yaw_error_deg':yawerrors,'tilt_deg':tilts,'grip_force':forces.tolist(),'table_support':support.tolist(),'peak_penetration_m':self.peak,'worst':self.worst,'saturation_substeps':self.saturation,'finite':bool(np.isfinite(d.qpos).all()),'warnings':[int(w.number) for w in d.warning], 'corners':corners,'frame_geometry_ok':bool(all(c['ok'] for c in corners) and max(errors)<.006 and max(yawerrors)<1 and max(tilts)<1 and np.max(abs(d.xpos[self.woods,2]-H/2))<.003)}
