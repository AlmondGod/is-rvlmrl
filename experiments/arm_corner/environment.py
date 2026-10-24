"""Three articulated xArm7 chains. Only motors act after reset; wood is free."""
import copy,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from experiments.box_assembly.design import beveled_vertices
from experiments.box_assembly.environment import fmt
SOURCE=Path(__file__).resolve().parents[2]/'third_party/3-PoinTr/simulation_assets/ufactory_xarm7'
SIZE=np.array([.038,.9144,.064]);BASES=[(.80,-.28,.02),(-.45,.92,.02),(.95,.90,.02)]
STARTS=[(.38,-.21,.032),(-.24,.37,.032)]

def scene(dt=.001):
 root=ET.Element('mujoco',model='articulated_marker_free_corner');ET.SubElement(root,'compiler',angle='radian',autolimits='true');o=ET.SubElement(root,'option',timestep=str(dt),integrator='implicitfast',cone='elliptic',iterations='100',tolerance='1e-9');ET.SubElement(o,'flag',multiccd='enable')
 v=ET.SubElement(root,'visual');ET.SubElement(v,'global',offwidth='1280',offheight='960');ET.SubElement(v,'headlight',ambient='.5 .5 .5',diffuse='.6 .6 .6');default=ET.SubElement(root,'default');asset=ET.SubElement(root,'asset');w=ET.SubElement(root,'worldbody');act=ET.SubElement(root,'actuator');con=ET.SubElement(root,'contact');eq=ET.SubElement(root,'equality');ten=ET.SubElement(root,'tendon')
 ET.SubElement(w,'light',pos='0 0 3',dir='0 0 -1');ET.SubElement(w,'geom',name='bench',type='box',size='1.25 1.2 .045',pos='0 .05 -.045',rgba='.12 .17 .20 1',friction='.5 .001 .0001',solref='.004 1',solimp='.99 .999 .001 .5 2')
 for i,(pos,look) in enumerate([((0,0,2.4),(0,0,0)),((1.45,-1.5,1.5),(0,.1,.1)),((-1.45,-1.0,1.5),(0,.1,.1))]):
  back=np.array(pos)-look;back=back/np.linalg.norm(back);right=np.cross([0,0,1],back)
  if np.linalg.norm(right)<1e-5:right=np.array([1.,0,0])
  right/=np.linalg.norm(right);up=np.cross(back,right);ET.SubElement(w,'camera',name=f'camera{i}',pos=fmt(pos),xyaxes=fmt(np.r_[right,up]),fovy='55')
 for i in range(3):
  r=ET.parse(SOURCE/'xarm7.xml').getroot();prefix=f'a{i}_'
  for mesh in r.findall('./asset/mesh'):
   mesh.set('name',mesh.get('name',Path(mesh.get('file')).stem));mesh.set('file',str(SOURCE/'assets'/mesh.get('file')))
  for element in r.iter():
   for key in ('name','class','childclass','mesh','material','joint','joint1','joint2','body1','body2','tendon'):
    if key in element.attrib:element.set(key,prefix+element.get(key))
  base=r.find('./worldbody/body');base.set('pos',fmt(BASES[i]))
  # Native hand is retained on arms0/1. Arm2 carries a physical mounted tool.
  if i==2:
   wrist=base.find(".//body[@name='a2_link7']");wrist.remove(wrist.find("body[@name='a2_xarm_gripper_base_link']"))
   tool=ET.SubElement(wrist,'body',name='a2_nailer',pos='0 0 .025');ET.SubElement(tool,'geom',name='nailer_body',type='box',size='.04 .065 .09',pos='0 0 .09',mass='1.8',rgba='.22 .30 .34 1');ET.SubElement(tool,'geom',name='nailer_nose',type='cylinder',size='.009 .025',pos='0 0 .205',mass='.1',rgba='.12 .12 .12 1');ET.SubElement(tool,'site',name='a2_link_tcp',pos='0 0 .23',size='.001',group='4')
   r.remove(r.find('equality'));r.remove(r.find('tendon'));r.remove(r.find('contact'));r.find('actuator').remove(r.find("./actuator/general[@name='a2_gripper']"))
  for geom_index,element in enumerate(base.iter('geom')):
   if not element.get('name'):element.set('name',prefix+'collision_'+str(geom_index))

   if 'finger_pad' in element.get('name',''):element.set('rgba','.15 .15 .15 1');element.set('solimp','.99 .999 .001 .5 2')
  for tag,dest in [('asset',asset),('worldbody',w),('actuator',act),('contact',con),('equality',eq),('tendon',ten)]:
   section=r.find(tag)
   if section is not None:
    for child in section:dest.append(child)
  for child in r.find('default'):default.append(child)
  ET.SubElement(con,'exclude',body1=prefix+'link_base',body2='world')
 for i,pos in enumerate(STARTS):
  name=f'wood{i}';ET.SubElement(asset,'mesh',name=name,vertex=fmt(np.array(beveled_vertices(SIZE,.0005)).ravel()));b=ET.SubElement(w,'body',name=name,pos=fmt(pos),quat=fmt([1,0,0,0] if i==0 else [2**-.5,0,0,2**-.5]));ET.SubElement(b,'freejoint',name=name+'_free');ET.SubElement(b,'geom',name=name+'_geom',mesh=name,type='mesh',density='600',rgba='.72 .49 .28 1',friction='.5 .001 .0001',solref='.004 1',solimp='.99 .999 .001 .5 2')
  for k in (0,1):
   for side in ('left','right'):
    for n in (1,2):ET.SubElement(con,'pair',geom1=name+'_geom',geom2=f'a{k}_{side}_finger_pad_{n}',condim='3',friction='.65 .65 .001 .0001 .0001',solref='.004 1',solimp='.99 .999 .001 .5 2')
 return ET.tostring(root,encoding='unicode')

def down_rotation(yaw):
 # TCP localx along rail's long axis; localz downward, fingers close along localy.
 a=np.array([-np.sin(yaw),np.cos(yaw),0]);b=np.cross([0,0,-1],a);return np.column_stack((a,b,[0,0,-1]))

class Env:
 def __init__(self,seed=0,dt=.001):
  self.xml=scene(dt);self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m);self.ikd=mujoco.MjData(self.m);m,d=self.m,self.d
  self.ids=[np.array([m.joint(f'a{i}_joint{k}').id for k in range(1,8)]) for i in range(3)];self.qids=[m.jnt_qposadr[j] for j in self.ids];self.vids=[m.jnt_dofadr[j] for j in self.ids];self.aids=[np.array([m.actuator(f'a{i}_act{k}').id for k in range(1,8)]) for i in range(3)];self.sites=[m.site(f'a{i}_link_tcp').id for i in range(3)];self.grips=[m.actuator(f'a{i}_gripper').id for i in (0,1)];self.woods=[m.body(f'wood{i}').id for i in (0,1)];self.targets=np.zeros((3,3));self.rotations=np.array([down_rotation(0),down_rotation(np.pi/2),down_rotation(0)])
  rng=np.random.default_rng(seed)
  for i in (0,1):
   qa=int(m.joint(f'wood{i}_free').qposadr[0]);d.qpos[qa:qa+2]+=rng.uniform(-.007,.007,2);yaw=(0 if i==0 else np.pi/2)+rng.uniform(-.04,.04);d.qpos[qa+3:qa+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
  for i in range(3):d.qpos[self.qids[i]]=[0,-.4,0,1.1,0,.6,0]
  mujoco.mj_forward(m,d)
  self.targets[:]=[[.38,-.21,.30],[-.24,.37,.30],[.9,.45,.4]]
  self.ikerrors=[]
  for i in range(3):
   q,err=self.ik(i,self.targets[i],self.rotations[i],iterations=250,multistart=True);d.qpos[self.qids[i]]=q;d.ctrl[self.aids[i]]=q;self.ikerrors.append(err)
  mujoco.mj_forward(m,d);self.qtarget=[d.qpos[q].copy() for q in self.qids];self.integral=np.zeros((3,7));self.gripper=np.zeros(2);self.peak=0.;self.worst={};self.saturation=0;self.contacts=[];self.velocity_limits=np.full(7,.7)
 def ik(self,arm,p,R,iterations=40,multistart=False):
  m=self.m;d=self.ikd;d.qpos[:]=self.d.qpos;qa=self.qids[arm];va=self.vids[arm];sid=self.sites[arm];best=(d.qpos[qa].copy(),1e9);jp=np.zeros((3,m.nv));jr=jp.copy();initial=d.qpos[qa].copy();rng=np.random.default_rng(123)
  for attempt in range(12 if multistart else 1):
   d.qpos[qa]=initial if attempt==0 else np.array([rng.uniform(-2,2),rng.uniform(-1,1),rng.uniform(-2,2),rng.uniform(.3,2.5),rng.uniform(-2,2),rng.uniform(-1,2),rng.uniform(-2,2)])
   for _ in range(iterations):
    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);cur=d.site_xmat[sid].reshape(3,3);e=np.r_[p-d.site_xpos[sid],.3*Rotation.from_matrix(R@cur.T).as_rotvec()];err=np.linalg.norm(e)
    if err<best[1]:best=(d.qpos[qa].copy(),float(err))
    if err<1e-5:break
    mujoco.mj_jacSite(m,d,jp,jr,sid);J=np.vstack((jp[:,va],.3*jr[:,va]));dq=J.T@np.linalg.solve(J@J.T+.0001*np.eye(6),e);d.qpos[qa]+=np.clip(dq,-.15,.15);d.qpos[qa]=np.clip(d.qpos[qa],m.jnt_range[self.ids[arm],0]+.01,m.jnt_range[self.ids[arm],1]-.01)
   if best[1]<1e-5:break
  return best
 def robot(self):
  return {'time':float(self.d.time),'tcp':self.d.site_xpos[self.sites].copy(),'rotation':self.d.site_xmat[self.sites].reshape(3,3,3).copy(),'joints':np.array([self.d.qpos[q] for q in self.qids]),'joint_velocity':np.array([self.d.qvel[v] for v in self.vids]),'gripper_command':self.gripper.copy()}
 def step(self,targets=None,rotations=None,gripper=None):
  m,d=self.m,self.d
  if targets is not None:
   for i in range(3):
    delta=np.asarray(targets[i])-self.targets[i];self.targets[i]+=delta*min(1.,.07*.02/max(np.linalg.norm(delta),1e-9))
  if rotations is not None:
   for i in range(3):
    rv=Rotation.from_matrix(rotations[i]@self.rotations[i].T).as_rotvec();self.rotations[i]=Rotation.from_rotvec(rv*min(1.,.3*.02/max(np.linalg.norm(rv),1e-9))).as_matrix()@self.rotations[i]
  if gripper is not None:self.gripper+=np.clip(np.asarray(gripper)-self.gripper,-2.5,2.5)
  errors=[]
  for i in range(3):
   q,err=self.ik(i,self.targets[i],self.rotations[i]);self.qtarget[i]+=np.clip(q-self.qtarget[i],-.7*.02,.7*.02);self.integral[i]=np.clip(self.integral[i]+200*.02*(self.qtarget[i]-d.qpos[self.qids[i]]),-8,8);d.ctrl[self.aids[i]]=self.qtarget[i]+(d.qfrc_bias[self.vids[i]]+self.integral[i])/m.actuator_gainprm[self.aids[i],0];errors.append(err)
  d.ctrl[self.grips]=self.gripper
  assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
  internal=getattr(self,'_joint_internal_forces',None)
  for _ in range(round(.02/m.opt.timestep)):
   assert not np.any(d.xfrc_applied)
   if internal is not None:
    d.qfrc_applied[:]=0;internal() # paired internal finite-strength connection loads only
   else:assert not np.any(d.qfrc_applied)
   mujoco.mj_step(m,d)
   for c in d.contact:
    if -c.dist>self.peak:self.peak=float(-c.dist);self.worst={'time':float(d.time),'pair':[m.geom(g).name or str(g) for g in (c.geom1,c.geom2)],'depth_m':self.peak}
   self.saturation+=int(any(abs(d.actuator_force[a])>=m.actuator_forcerange[a,1]-.01 for ids in self.aids for a in ids))
  mujoco.mj_forward(m,d)
  # mj_step contacts describe its evaluated state; include the final forwarded
  # configuration too so the last substep cannot escape the penetration gate.
  for c in d.contact:
   if -c.dist>self.peak:self.peak=float(-c.dist);self.worst={'time':float(d.time),'pair':[m.geom(g).name or str(g) for g in (c.geom1,c.geom2)],'depth_m':self.peak}
  if internal is not None:d.qfrc_applied[:]=0
  return errors
 def metrics(self):
  m,d=self.m,self.d;poses=[];forces=np.zeros(2);support=np.zeros(2);self.contacts=[];f=np.zeros(6)
  for k,c in enumerate(d.contact):
   mujoco.mj_contactForce(m,d,k,f);names=[m.geom(g).name or '' for g in (c.geom1,c.geom2)];self.contacts.append({'names':names,'distance':float(c.dist),'force':f.tolist()})
   for i in (0,1):
    if f'wood{i}_geom' in names:
     if any('finger_pad' in n for n in names):forces[i]+=max(0,f[0])
     elif 'bench' in names:support[i]+=max(0,f[0])
  for b in self.woods:poses.append({'center':d.xpos[b].tolist(),'rotation':d.xmat[b].reshape(3,3).tolist()})
  p0,p1=[d.xpos[b] for b in self.woods];R0,R1=[d.xmat[b].reshape(3,3) for b in self.woods];u0=R0[:,1];u1=-R1[:,1] # reference yaw+pi/2 points negative x; choose positive x
  goal=p1+u1*(SIZE[1]/2-SIZE[0]/2)-u0*(SIZE[1]/2+SIZE[0]/2)
  return {'wood_evaluator_only':poses,'grip_force':forces.tolist(),'table_support':support.tolist(),'corner_position_error_m':float(np.linalg.norm(p0-goal)),'square_error_deg':float(np.degrees(np.arcsin(np.clip(abs(np.dot(u0,u1)),0,1)))),'peak_penetration_m':self.peak,'worst':self.worst,'saturation_substeps':self.saturation,'finite':bool(np.isfinite(d.qpos).all()),'warnings':[int(w.number) for w in d.warning]}
 def images(self,renderer):
  frames=[];cameras=[];h,w=renderer.height,renderer.width
  for i in range(3):
   renderer.update_scene(self.d,camera=f'camera{i}');frames.append(renderer.render().copy());c=self.m.camera(f'camera{i}').id;R=np.diag([1.,-1.,-1.])@self.d.cam_xmat[c].reshape(3,3).T;T=np.eye(4);T[:3,:3]=R;T[:3,3]=-R@self.d.cam_xpos[c];f=.5*h/np.tan(np.deg2rad(self.m.cam_fovy[c])/2);cameras.append({'K':[[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]],'world_to_camera':T.tolist()})
  return np.array(frames),cameras
