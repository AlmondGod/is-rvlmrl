"""Full free-body box server. Controller only gets RGB/calibration/robot state.

A separate Astra agent chooses actions online. The physics server is paused
between action chunks. This is not real-time VLM control and not a security sandbox.
"""
import argparse,json,hashlib,time
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
import imageio.v2 as imageio
import cv2
import mujoco
import numpy as np
from .design import Config,parts
from .environment import fmt
from .fastening_environment import FasteningBoxAssembly,tool_xml


def xml(config):
 root=ET.fromstring(tool_xml(config));w=root.find('worldbody')
 for name,pos,look in [('overhead',(0,0,1.35),(0,0,0)),('opposite',(-.75,-.65,.70),(0,0,.08))]:
  back=np.array(pos)-look;back=back/np.linalg.norm(back)
  right=np.cross([0.,1.,0.] if name=='overhead' else [0.,0.,1.],back);right/=np.linalg.norm(right);up=np.cross(back,right)
  ET.SubElement(w,'camera',name=name,pos=fmt(pos),xyaxes=fmt(np.r_[right,up]),fovy='60')
 for axis in (0,1):
  for v in np.arange(-.6,.61,.1):
   if axis==0:pos=[v,0,.00005];size=[.0002,.54,.00005]
   else:
    if abs(v)>.5:continue
    pos=[0,v,.00005];size=[.67,.0002,.00005]
   ET.SubElement(w,'geom',name=f'grid_{axis}_{v:.1f}',type='box',pos=fmt(pos),size=fmt(size),rgba='.30 .34 .38 1',contype='0',conaffinity='0')
 return ET.tostring(root,encoding='unicode')


def clean(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 return x


def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean(value),indent=2)+'\n');tmp.replace(path)


class Bridge:
 def __init__(self,out,seed,timestep=.0005):
  self.out=out;self.public=out/'controller';self.private=out/'evaluator';self.public.mkdir(parents=True,exist_ok=False);self.private.mkdir()
  (self.public/'requests').mkdir();(self.public/'replies').mkdir()
  self.e=FasteningBoxAssembly(replace(Config(),timestep=timestep),allow_local=True,xml_builder=xml);self.e.reset(seed=seed)
  self.renderer=mujoco.Renderer(self.e.model,height=480,width=640)
  self.option=mujoco.MjvOption();self.option.geomgroup[2]=0;self.option.sitegroup[:]=0
  self.writer=imageio.get_writer(out/'astra_camera_attempt.mp4',fps=10,codec='libx264',quality=8,macro_block_size=1)
  self.index=0;self.stop=False;self.camera_names=('vision_0','vision_1','overhead','opposite');self.wall_start=time.monotonic()
  # Settle without manipulating parts. This is logged evaluator-side, not a reset to assembled geometry.
  for _ in range(50):self.e.step(self.e.wrist_target.copy(),self.e.wrist_quaternion.copy(),self.e.finger_target.copy())
  self.pose={'position':self.e.wrist_target.copy(),'quaternion':self.e.wrist_quaternion.copy(),'fingers':self.e.finger_target.copy()}
  self.tools={};self.configuration();self.snapshot()
  save(self.private/'initial.json',self.e.evaluator_metrics())

 def views(self):
  images=[];cal=[];m,d=self.e.model,self.e.data
  for name in self.camera_names:
   self.renderer.update_scene(d,camera=name,scene_option=self.option);self.e._render_extra(self.renderer.scene)
   images.append(self.renderer.render().copy());cid=m.camera(name).id
   R=np.diag([1.,-1.,-1.])@d.cam_xmat[cid].reshape(3,3).T;T=np.eye(4);T[:3,:3]=R;T[:3,3]=-R@d.cam_xpos[cid]
   f=.5*480/np.tan(np.deg2rad(m.cam_fovy[cid])/2)
   cal.append({'name':name,'K':[[f,0,319.5],[0,f,239.5],[0,0,1]],'world_to_camera':T})
  return images,cal

 def configuration(self):
  e=self.e;m=e.model
  config={'task':'Assemble five separately staged wooden panels into an open-top box, using camera feedback. Nail/glue operation currently implemented at one front-left joint only. Complete-box fastening cannot be claimed.',
   'observation_contract':'RGB + camera calibration + robot/tool proprioception only. No object states, contacts, reward or evaluator metrics.',
   'world_coordinates':'meters; X/Y horizontal, Z up. Bench surface Z=0. Grid spacing=0.10m.',
   'design_targets':{n:{'size_m':p['size'],'assembled_center_m':p['target'],'assembled_quaternion_wxyz':[1,0,0,0]} for n,p in parts(e.config).items()},
   'robot':'Wuji Hand 2.2 right, 20 finger motors; virtual force/torque-limited wrist carrier; fingers extend along local -Z at zero.',
   'finger_names':[m.joint(j).name for j in e.finger_joint_ids],'finger_limits_rad':m.actuator_ctrlrange[e.finger_actuators],
   'action_schema':{'duration_s':'0.02..2.0, simulation pauses afterward','position':'absolute wrist XYZ meters (optional)','quaternion':'absolute wrist quaternion wxyz (optional)',
    'fingers':'20 joint targets radians (optional)','fixture_release':'boolean, retract true/engage false; moves physical jaws',
    'tools':'optional mapping nailer/glue_tool to position/quaternion/trigger/dispense',
    'reason':'required explanation based on observed image/proprioception','stop':'optional boolean to end run and finalize video'},
   'simulation':{'integration_step_s':e.config.timestep,'control_period_s':e.config.control_period,'pauses_between_commands':True},
   'limits':{'wrist_position_m':[[-.60,-.48,.025],[.60,.48,.65]],'wrist_speed_m_s':.15,'wrist_rotation_rad_s':1.,'force_N':35,'torque_Nm':3},
   'fixture':'Four retractable corner carriers; both interior pads and exterior stops lift when released. Place base with fixture retracted; avoid lowered pads during placement.',
   'static_prior_disclosure':'Panel dimensions and desired assembly positions are blueprint information, not measured current panel poses.',
   'no_reset':'No reset, grasp attachment, object teleport or privileged pick primitive is available.'}
  save(self.public/'config.json',config)

 def snapshot(self):
  e=self.e;m,d=e.model,e.data;images,cameras=self.views()
  paths=[]
  for name,im in zip(self.camera_names,images):
   path=self.public/f'{self.index:04d}_{name}.png';imageio.imwrite(path,im);paths.append(str(path.resolve()))
  # Only camera imagery and robot-specific kinematics are constructed here.
  robot={'wrist_position':d.xpos[e.wid].copy(),'wrist_quaternion':d.xquat[e.wid].copy(),
   'finger_q':d.qpos[e.finger_qadr].copy(),'finger_qvel':d.qvel[e.finger_vadr].copy(),
   'wrist_target':e.wrist_target.copy(),'wrist_target_quaternion':e.wrist_quaternion.copy(),'finger_targets':e.finger_target.copy(),
   'fingertip_positions':{m.site(i).name:d.site_xpos[i].copy() for i in range(m.nsite) if m.site(i).name.startswith('r_') and 'tip' in m.site(i).name},
   'fixture_joint_positions':d.qpos[m.jnt_qposadr[e.fixture_joints]].copy()}
  packet={'schema':'box_camera_online_v1','observation_index':self.index,'simulation_time':float(d.time),
   'images':dict(zip(self.camera_names,paths)),'cameras':cameras,'robot':robot,'tools':e.tool_observation(),
   'config_path':str((self.public/'config.json').resolve())}
  assert set(packet)=={'schema','observation_index','simulation_time','images','cameras','robot','tools','config_path'}
  save(self.public/f'{self.index:04d}_observation.json',packet);save(self.public/'latest.json',packet)
  self.packet=packet
  return packet

 def record(self):
  images,_=self.views();frame=np.concatenate([np.concatenate(images[:2],axis=1),np.concatenate(images[2:],axis=1)],axis=0)
  cv2.rectangle(frame,(0,0),(1279,44),(14,22,29),-1)
  cv2.putText(frame,f'ASTRA ONLINE | RGB + ROBOT STATE | action {self.index} | sim {self.e.data.time:.2f}s',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(225,235,245),1,cv2.LINE_AA)
  self.writer.append_data(frame)

 def action(self,action):
  allowed={'duration_s','position','quaternion','fingers','fixture_release','tools','reason','stop'}
  if set(action)-allowed or not isinstance(action.get('reason'),str):raise ValueError('Unknown fields or missing reason')
  if action.get('stop'):self.stop=True;return {'stopped':True,'observation':self.packet}
  duration=float(action.get('duration_s',.5))
  if not np.isfinite(duration) or not .02<=duration<=2.:raise ValueError('duration_s must be .02..2.0')
  for key,size in [('position',3),('quaternion',4),('fingers',20)]:
   if key in action:
    v=np.asarray(action[key],float)
    if v.shape!=(size,) or not np.isfinite(v).all():raise ValueError('Invalid '+key)
    self.pose[key]=v
  if 'fixture_release' in action:self.e.fixture_release(bool(action['fixture_release']))
  tools=action.get('tools',{})
  steps=max(1,round(duration/self.e.config.control_period));self.index+=1
  for i in range(steps):
   metrics=self.e.step(self.pose['position'],self.pose['quaternion'],self.pose['fingers'],tools=tools)
   if i%5==0:self.record()
   if not metrics['finite_state'] or any(metrics['warnings']):raise RuntimeError('Numerical failure; rollout stopped')
   if metrics['peak_episode_penetration_m']>.005:raise RuntimeError('Excessive penetration; rollout stopped')
  save(self.private/f'{self.index:04d}_metrics.json',metrics)
  return {'observation':self.snapshot()}

 def run(self):
  print(json.dumps({'ready':str(self.public.resolve())}),flush=True)
  try:
   while not self.stop:
    requests=sorted((self.public/'requests').glob('*.json'),key=lambda p:p.stat().st_mtime_ns)
    if not requests:time.sleep(.1);continue
    for p in requests:
     request=json.loads(p.read_text());token=request['request_id'];start=time.monotonic()
     save(self.private/f'request_{token}.json',request)
     try:response=self.action(request['action'])
     except Exception as exc:
      response={'error':type(exc).__name__+': '+str(exc)}
      if isinstance(exc,RuntimeError):self.stop=True
     response['execution_wall_seconds']=time.monotonic()-start
     save(self.public/'replies'/f'{token}.json',response);p.unlink()
     if self.stop:break
  finally:
   self.writer.close();self.renderer.close()
   save(self.private/'final.json',{'metrics':self.e.evaluator_metrics(),'wall_seconds':time.monotonic()-self.wall_start,
       'actions':self.index,'policy_model':'gpt-6-astra','online_model_decisions':True,'privileged_object_state_to_controller':False})


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=7);p.add_argument('--timestep',type=float,default=.0005);a=p.parse_args()
 a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);Bridge(a.out,a.seed,a.timestep).run()

if __name__=='__main__':main()
