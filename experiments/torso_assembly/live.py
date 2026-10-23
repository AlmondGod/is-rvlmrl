"""Online external Astra decisions, RGB/encoder skills, private evaluator.

The controller directory is a data-access contract, not an OS security sandbox.
No oracle branch, diagnostic checkpoint, or fixed full-task action sequence.
"""
import argparse,json,time,shutil,hashlib
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,cv2
from experiments.arm_corner.run import clean
from experiments.wall_panel.perception import Vision as PanelVision,HeldSheet
from .environment import Env,GOALS,SIZES,PANEL_GOAL,PANEL_SIZE,OFFSETS,SIDE_R
from .frame_skills import Skills as FrameSkills,visual_frame_ok
from .frame_vision import MultiVision,probe_anchor
from .panel_skills import Skills as PanelSkills
from .planner import Planner
from .tool_skills import Tool

def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean(value),indent=2));tmp.replace(path)

def packet(index,robot,phase,command_id,failure,rgb,cameras,events,completed):
 # Whitelist: no object poses, estimator output, contacts, rewards or evaluator.
 robot={k:robot[k] for k in ('time','tcp','rotation','joints','joint_velocity','gripper_command','shuttle_x','rack_slide0','rack_slide1')}
 events=[{k:v for k,v in event.items() if k in ('kind','index','time','tcp','abstract','triggered')} for event in events]
 return dict(observation_index=index,sim_time_s=robot['time'],phase=phase,
             last_command_id=command_id,last_failure=failure,robot=robot,
             rgb=rgb,camera_calibration=cameras,abstract_fasteners=events,
             completed_commands=completed)

class Executor:
 def __init__(self,e,tool_factory=Tool):
  self.tool_factory=tool_factory
  self.e=e;self.planner=Planner(e);self.frame=FrameSkills(e.robot(),self.planner)
  self.panel=None;self.tool=None;self.flip=None;self.flipped=False;self.held=HeldSheet();self.phase='idle';self.failure=None
  self.queue=[];self.kind=None;self.timer=0.;self.events=[];self.placed=set();self.released=set()
  self.panel_fed=False;self.panel_placed=False;self.panel_released=False;self.planning=[]
  self.cmd=(e.targets.copy(),e.rotations.copy(),e.gripper.copy(),None)
 def start(self,c,r,est,sheet):
  if self.phase not in ('idle','done'):raise ValueError('Command while busy or failed')
  skill=c['skill'];i=c['index']
  if skill in ('position_member','release_member','nail_corner','nail_panel') and (type(i) is not int or not 0<=i<4):raise ValueError('index must be 0..3')
  if skill=='position_member':
   if i in self.placed:raise ValueError('Member already positioned')
   self.queue=[('position',i)]
  elif skill=='release_member':
   if i not in self.placed or i in self.released:raise ValueError('Member is not awaiting release')
   self.queue=[('release',i)]
  elif skill=='nail_corner':
   if len(self.released)!=4 or not visual_frame_ok(est):raise ValueError('Need four released visually aligned members')
   if any(x['kind']=='frame' and x['index']==i for x in self.events):raise ValueError('Duplicate corner nail')
   self.queue=[('frame_nail',i)]
  elif skill=='feed_panel':
   if self.panel_fed or sum(x['kind']=='frame' for x in self.events)!=4:raise ValueError('Need four corner visits before sheet feed')
   self.queue=[('feed',None)]
  elif skill=='position_panel':
   if not self.panel_fed or self.panel_placed:raise ValueError('Sheet must be fed once before placement')
   self.queue=[('panel_prepare',0),('panel_prepare',1),('panel_position',None)]
  elif skill=='release_panel':
   if not self.panel_placed or self.panel_released:raise ValueError('Sheet is not awaiting release')
   self.queue=[('panel_release',None)]
  elif skill=='nail_panel':
   if not self.panel_released:raise ValueError('Release sheet first')
   if any(x['kind']=='panel' and x['index']==i for x in self.events):raise ValueError('Duplicate panel nail')
   self.queue=[('panel_nail',i)]
  elif skill=='flip_structure':
   if not getattr(self.e,'flip_enabled',False):raise ValueError('Flip is not enabled for this fixture/layout; restart with --flip')
   if not self.panel_released or len(self.events)!=8 or self.e.nails is None:raise ValueError('Flip needs a released sheet and eight nail triggers with fastener physics enabled')
   self.queue=[('flip',None)]
  else:raise ValueError('Unknown skill')
  self.kind=None;self.phase='starting';self.advance(r,est,sheet)
 def advance(self,r,est,sheet):
  if not self.queue:self.phase='done';self.kind=None;return
  self.kind,self.i=self.queue.pop(0);i=self.i;self.timer=0.;started=time.monotonic()
  if self.kind in ('position','release'):
   self.frame.start({'skill':self.kind,'piece':i},est,r);self.frame.lift_height=.25
  elif self.kind.endswith('nail'):self.tool=self.tool_factory(r,self.planner,'frame' if self.kind=='frame_nail' else 'panel',i,est,sheet)
  elif self.kind=='panel_prepare':
   if sheet is None:raise ValueError('No sheet RGB estimate')
   target=np.array(sheet['center'])+OFFSETS[i];target[2]=sheet['center'][2];target[0]+=(-.12 if i==0 else .12)
   self.route=np.array(self.planner.plan(i,r,est,target,SIDE_R[i],sheet));self.route_index=1
  elif self.kind=='panel_position':self.panel=PanelSkills(r,self.planner,est);self.panel.start('position',None,r,sheet)
  elif self.kind=='panel_release':self.panel.start('release',None,r,sheet)
  elif self.kind=='flip':
   from .flip import Flip
   self.flip=Flip(r,self.planner,est,sheet,**getattr(self.e,'flip_options',{}))
  self.planning.append(dict(kind=self.kind,wall_s=time.monotonic()-started));self.phase=self.kind
 def command(self,r,est,visual):
  if self.kind is None:return self.cmd
  self.timer+=.02;i=self.i;done=False
  sheet=self.held.estimate(visual,r) if self.panel and self.panel.holding and self.held.reference is not None and self.panel.state!='release' else visual
  if self.kind=='flip':
   self.cmd=self.flip.command(r);self.phase='flip: '+self.flip.state;done=self.flip.state=='done'
   if done:self.flipped=True
  elif self.kind in ('position','release'):
   self.cmd=self.frame.command(r,est);self.phase=f'{self.kind} member {i}: {self.frame.state}'
   if self.frame.state=='failed':raise RuntimeError(self.frame.failure)
   done=self.frame.state=='done'
   if done:(self.placed if self.kind=='position' else self.released).add(i)
  elif self.kind.endswith('nail'):
   self.cmd=self.tool.command(r);self.phase=f'{self.kind} {i}: {self.tool.state}';done=self.tool.state=='done'
   if self.e.nails is not None and self.tool.events and not getattr(self.tool,'shot_applied',False):
    self.e.nails.fire(self.tool.kind,i);self.tool.shot_applied=True
    self.tool.events[0].update(abstract=False,physical_bond=None,triggered=True)
   if done:self.events+=self.tool.events;self.cmd=(r['tcp'].copy(),r['rotation'].copy(),np.zeros(2),None)
  elif self.kind=='feed':
   self.e.shuttle_command=max(0.,self.e.shuttle_command-.0006)
   self.cmd=(r['tcp'].copy(),r['rotation'].copy(),np.zeros(2),{a:r['joints'][a] for a in range(3)})
   done=r['shuttle_x']<.001 and self.timer>47
   if self.timer>65:raise RuntimeError('shuttle_timeout')
   if done:self.panel_fed=True
  elif self.kind=='panel_prepare':
   self.cmd=(r['tcp'].copy(),r['rotation'].copy(),np.zeros(2),{i:self.route[self.route_index]})
   if np.max(abs(r['joints'][i]-self.route[self.route_index]))<.01:self.route_index+=1
   done=self.route_index>=len(self.route)
   if self.timer>45:raise RuntimeError('panel_approach_timeout')
  else:
   self.cmd=self.panel.command(r,sheet,visual);self.phase='panel: '+self.panel.state;self.e.rack_command=self.panel.rack
   if self.panel.begin_grasp:self.held.begin(sheet,r);self.panel.begin_grasp=False
   done=self.panel.state=='done'
   if done:
    if self.kind=='panel_position':self.panel_placed=True
    else:self.panel_released=True
  if done:self.advance(r,est,sheet)
  return self.cmd

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--layout-seed',type=int);ap.add_argument('--physical-nails',action='store_true');ap.add_argument('--flip',action='store_true');ap.add_argument('--flip-profile',choices=['legacy','clearance-stack'],default='clearance-stack');ap.add_argument('--shoulder-half-span',type=float);ap.add_argument('--dt',type=float);ap.add_argument('--seconds',type=float,default=1200);a=ap.parse_args()
 if a.dt is None:a.dt=.0005 if a.flip and a.flip_profile=='clearance-stack' else .001
 if a.shoulder_half_span is None:a.shoulder_half_span=.55 if a.flip and a.flip_profile=='clearance-stack' else .34
 if a.flip and not a.physical_nails:raise ValueError('--flip requires --physical-nails')
 out=Path(a.out).resolve();out.mkdir(exist_ok=False,parents=True);public=out/'controller';public.mkdir();private=out/'evaluator';private.mkdir()
 for name in ('torso_assembly','frame_assembly_v2','wall_panel','arm_corner','astra_camera','box_assembly'):shutil.copytree('experiments/'+name,private/'source'/name,ignore=shutil.ignore_patterns('__pycache__'))
 save(private/'source_hashes.json',{str(p.relative_to(private/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (private/'source').rglob('*') if p.is_file()})
 save(private/'experiment.json',{'layout_seed_evaluator_only':a.layout_seed,'distribution':'face-up, surveyed support height; within-side lane permutation; XY jitter ±12/35 mm, yaw ±0.06 rad','privileged_workpiece_poses_in_controller':False})
 e=Env(dt=a.dt,layout_seed=a.layout_seed,physical_nails=a.physical_nails,fixture_half_width=.38 if a.flip and a.flip_profile=='clearance-stack' else .46,shoulder_half_span=a.shoulder_half_span);e.flip_enabled=a.flip;e.flip_options=dict(grasp='stack',smooth=True,clearance=True) if a.flip_profile=='clearance-stack' else {};ex=Executor(e);fv=MultiVision();pv=PanelVision();ren=mujoco.Renderer(e.m,960,1280);writer=io.get_writer(out/'continuous.mp4',fps=10,codec='libx264',quality=7)
 def robot():
  r=e.robot()
  for name in ('shuttle_x','rack_slide0','rack_slide1'):r[name]=float(e.d.qpos[int(e.m.joint(name).qposadr[0])])
  return r
 instructions={'task':'From loose side-staged lumber, assemble a 0.9144 m square frame, visit each corner once with the nailgun, feed/place/release the plywood sheet, and visit each panel-to-member fastening point once. Three UR10e arms share one fixed torso. No colored markers.',
 'access':'Read ONLY this controller directory and referenced RGB images. Do not inspect source, evaluator, prior runs, other agents or simulator state. Do not edit code or generate a fixed sequence. This is a data contract, not an OS sandbox.',
 'sensors':'RGB images, camera calibration, robot/fixture encoders, command status and abstract tool-event history. No object state or evaluator metrics. Use the images before EACH command. Physics continues during your reasoning, subject to measured compute lag.',
 'blueprint':{'centers_m':GOALS.tolist(),'member_sizes_m':SIZES.tolist(),'panel_size_m':PANEL_SIZE.tolist(),'panel_goal_m':PANEL_GOAL.tolist()},
 'member_identity':'Locations and yaw are unknown. Perception globally discovers the four boards from RGB and assigns roles by observed side and length:0 long left-side member to front;1 short left-side member to left;2 long right-side member to rear;3 short right-side member to right. Arm0 handles left-side candidates, arm1 right-side candidates, arm2 nailgun. Lane order is not known.',
 'commands':'Write command.json atomically, exactly {command_id,observed_index,skill,index,reason}. Consecutive command_id starts1. index is0..3 for member/corner/panel-nail commands, null otherwise. observed_index refers to the RGB snapshot you actually inspected and must be no more than30 observations old. One command at a time, phase idle/done only. On failed primitive choose stop. Always inspect fresh images after completion before choosing the next action.',
 'skills':{'position_member':'Pick specified member, visually verify lift, transport and align to blueprint using RGB/encoders. Does not release.','release_member':'Open and park after visual acceptance.','nail_corner':'One abstract tool visit:0SW,1SE,2NE,3NW; needs four released visually aligned members.','feed_panel':'Move physical plywood feeder into workspace after four corner visits.','position_panel':'Prepare both grippers, grasp sheet from RGB outline, lift, withdraw shelves, align and lower; does not release.','release_panel':'Release and withdraw both grippers.','nail_panel':'One abstract fastening visit on member0..3, after sheet release.','finish':'Declare visual task completion and end.','stop':'End on observed failure or inability to safely continue.'},
 'limits':'You choose actions, order, visual acceptance and stopping. Skills are hand-written feedback controllers, not Astra motor weights. RGB geometry assumes known dimensions, calibration, feeder regions and no hidden slip after verified grasp. Primitive done is not evaluator success. No automatic full-task sequence, no hidden object attachments, no physical nail bonds or real-world validation.'}
 if a.physical_nails:
  instructions['fastening_model']='Finite-strength simulated point connections; instantaneous ray-gated insertion, no calibrated nail driving. abstract_fasteners is a legacy field name: entries report triggers, not latent bond condition.'
  instructions['limits']=instructions['limits'].replace('no physical nail bonds','provisional breakable nail connections')
 if a.flip:
  instructions['skills']['flip_structure']='After all eight nail triggers and sheet release, regrasp, lift and attempt a180-degree turn with five-second inverted hold. Stop on skill failure. This is a scripted RGB-initialized/encoder-controlled skill. Development simulation checks do not establish real-world or Astra-policy reliability.'
 if a.flip:instructions['task']+=' Finally perform flip_structure and inspect the inverted assembly before finish.'
 save(public/'instructions.json',instructions)
 flip_reference=None;pre_flip=None;flip_measurement=None
 trace=[];raw=[];actions=[];est=[None]*4;visual=None;obs=0;lastid=0;finished=False;error=None;maxlag=0.;lastphase='';completed=[];started=time.monotonic();stable=0.
 try:
  for step in range(round(a.seconds/.02)):
   r=robot()
   if step%5==0:
    frames,cameras=e.images(ren)
    if not ex.panel_fed:est=fv.update(frames,cameras,r,ex.frame.holding,probe_anchor(ex.frame))
    if r['shuttle_x']<.03:visual=pv.update(frames,cameras)
    ren.update_scene(e.d,camera='torso_overview');view=ren.render().copy();display=view.copy();cv2.rectangle(display,(0,815),(1280,960),(15,15,15),-1)
    for j,line in enumerate(['ASTRA RGB TASK CONTROL | '+ex.phase,f'{e.d.time:.1f}s | abstract tool visits {len(ex.events)}/8','Handwritten feedback skills | no runtime object state']):cv2.putText(display,line,(15,850+35*j),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),2)
    writer.append_data(display)
    if step%50==0:
     paths={}
     for k in (0,2,3):
      p=public/f'{obs:04d}_camera{k}.png';io.imwrite(p,frames[k]);paths[f'camera{k}']=str(p)
     p=public/f'{obs:04d}_workspace.png';io.imwrite(p,view);paths['workspace']=str(p)
     snap=packet(obs,r,ex.phase,lastid,ex.failure,paths,cameras,ex.events,completed);snap['published_unix_s']=time.time();save(public/f'{obs:04d}.json',snap);save(public/'latest.json',snap);obs+=1
   f=public/'command.json'
   if f.exists():
    c=json.loads(f.read_text())
    if c.get('command_id',0)>lastid:
     if set(c)!={'command_id','observed_index','skill','index','reason'} or c['command_id']!=lastid+1:raise ValueError('Invalid command schema/sequence')
     if type(c['observed_index']) is not int or not 0<=obs-1-c['observed_index']<=30:raise ValueError('Stale/invalid observation')
     if not isinstance(c['reason'],str) or not c['reason'].strip():raise ValueError('Missing image-based reason')
     if ex.phase not in ('idle','done','failed'):raise ValueError('Concurrent command')
     lastid+=1;source=public/f"{c['observed_index']:04d}.json";observed=json.loads(source.read_text());rec={'command':c,'received_sim_s':e.d.time,'received_wall_s':time.monotonic()-started,'observation_to_command_wall_s':time.time()-observed['published_unix_s']};actions.append(rec);save(private/'received_commands.json',actions)
     if c['skill'] in ('finish','stop'):finished=c['skill']=='finish';break
     ex.start(c,r,est,visual)
   if ex.kind=='flip' and flip_reference is None:
    from .flip_evaluator import reference
    flip_reference=reference(e);pre_flip=e.panel_metrics()
   before=ex.phase;cmd=ex.command(r,est,visual)
   if before!='done' and ex.phase=='done':completed.append({'command_id':lastid,'time_s':e.d.time})
   e.step(*cmd);m=e.panel_metrics();stable=stable+.02 if m['frame']['frame_geometry_ok'] and max(m['forces']['grip'])<.1 and m['forces']['frame']>50 and m['xy_error_m']<.006 else 0.
   if flip_reference is not None:
    from .flip_evaluator import measure
    flip_measurement=measure(e,flip_reference);m['flip']=flip_measurement
   if e.nails is not None:m['fasteners']=e.nails.metrics()
   trace.append({'time':e.d.time,'stage':ex.phase,'robot':clean(robot()),'est':clean(est),'visual':clean(visual),'vision_diagnostics':clean(fv.diagnostics),'metrics':m});raw.append(np.r_[e.d.time,e.d.qpos,e.d.qvel,e.d.ctrl,e.d.actuator_force])
   if ex.phase!=lastphase:print(round(e.d.time,2),ex.phase,flush=True);lastphase=ex.phase
   if e.peak>.0005 or any(m['frame']['warnings']) or not m['frame']['finite']:raise RuntimeError('Numerical gate:'+str(e.worst))
   delay=started+(step+1)*.02-time.monotonic()
   if delay>0:time.sleep(delay)
   else:maxlag=max(maxlag,-delay)
  else:error='deadline'
 except KeyboardInterrupt:error='interrupted'
 except Exception as exc:error=repr(exc);ex.failure=error;ex.phase='failed';print(error,flush=True)
 finally:
  writer.close();ren.close();save(private/'trace.json',trace);np.savez_compressed(private/'raw.npz',rows=raw,nq=e.m.nq,nv=e.m.nv,nu=e.m.nu);(private/'scene.xml').write_text(e.xml);m=e.panel_metrics()
  success=finished and error is None and len(ex.events)==8 and stable>=3 and m['frame']['frame_geometry_ok'] and m['xy_error_m']<.006 and abs(m['height_error_m'])<.003 and m['tilt_deg']<1
  if a.flip or flip_reference is not None:
   from .flip_evaluator import passes
   success=bool(finished and error is None and pre_flip is not None and pre_flip['frame']['frame_geometry_ok'] and pre_flip['xy_error_m']<.006 and flip_measurement is not None and passes(flip_measurement,e.nails.metrics()['intact'],ex.flipped))
  actual=bool(actions and (public/'model_connection.json').exists())
  save(private/'result.json',{'success':bool(success),'finished':finished,'error':error,'metrics':m,'events':ex.events,'actions':actions,'actual_astra':actual,'model':'gpt-6-astra' if actual else None,'privileged_diagnostic':False,'continuous':True,'diagnostic_checkpoint':None,'simulated_s':e.d.time,'wall_s':time.monotonic()-started,'max_wall_lag_s':maxlag,'physics_paused_for_model':False,'synchronous_planning_wall_s':ex.planning,'physical_validation':False,'no_runpod':True,'flip_requested':a.flip,'dt':a.dt,'flip_profile':a.flip_profile,'shoulder_half_span_m':a.shoulder_half_span,'flip_completed':ex.flipped,'flip_metrics':flip_measurement,'pre_flip_assembly':pre_flip,'fastener_physics':None if e.nails is None else e.nails.metrics()})
  save(public/'trial_closed.json',{'reason':'finished' if finished else ('controller_failure' if error else 'stopped'),'last_command_id':lastid,'evaluator_results':'private'})
  print('saved',out,flush=True)
if __name__=='__main__':main()
