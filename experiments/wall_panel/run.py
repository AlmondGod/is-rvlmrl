"""Camera baseline or external actual Astra decisions, with private evaluator logs."""
import argparse,json,time,shutil,hashlib
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,cv2
from .environment import Env,PANEL_GOAL,PANEL_SIZE
from .perception import Vision,HeldSheet
from .skills import Skills
from .tool_plan import PanelPlanner
from experiments.arm_corner.run import clean

def save(path,value):
 p=path.with_suffix('.tmp');p.write_text(json.dumps(clean(value),indent=2));p.replace(path)
def main(env_cls=Env,extra_sources=(),display_camera=None):
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--live',action='store_true');ap.add_argument('--seconds',type=float,default=600);ap.add_argument('--dt',type=float,default=.001);ap.add_argument('--seed',type=int,default=0);a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);public=out/'controller';public.mkdir();private=out/'evaluator';private.mkdir()
 for folder in ('wall_panel','frame_assembly_v2','arm_corner')+tuple(extra_sources):shutil.copytree('experiments/'+folder,private/'source'/folder,ignore=shutil.ignore_patterns('__pycache__'))
 save(private/'hashes.json',{str(p.relative_to(private/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (private/'source').rglob('*') if p.is_file()})
 offset=(0,0) if a.seed==0 else np.random.default_rng(a.seed).uniform(-.003,.003,2)
 e=env_cls(dt=a.dt,offset=offset);ren=mujoco.Renderer(e.m,960,1280);v=Vision();held=HeldSheet();writer=io.get_writer(out/'wall_panel.mp4',fps=10,codec='libx264',quality=7)
 def robot():
  r=e.robot();r['carriage']=float(e.d.qpos[int(e.m.joint('overhead_z').qposadr[0])]);r['rack']=float(e.d.qpos[int(e.m.joint('rack_slide0').qposadr[0])]);return r
 s=Skills(robot(),PanelPlanner(e));est=None;visual=None;trace=[];raw=[];actions=[];idx=0;lastid=0;started=time.monotonic();maxlag=0.;error=None;finished=False;missing=0.;lastphase='';settled=0.
 sequence=[('position',None),('release',None)]+[('nail',i) for i in range(4)];action=0
 save(public/'instructions.json',{'embodiment':getattr(e,'embodiment','Separate fixed robot mounts'),'task':'Place the single flat plywood sheet over the already assembled four-member frame, release it, then use the articulated nailgun for one abstract panel-to-frame nail on each member. This is a separate panel station initialized from the earlier completed frame, on a 120mm raised support. The staging shelves are visible and retract after lift. The nailgun is mounted overhead on a physical vertical slide.','allowed_directory':'ONLY this controller directory and its referenced RGB images. No source, evaluator, result files, or other runs.','sensors':'Three calibrated RGB views, robot encoders, status, logical fastening counter. No object ground truth. Physics continues while you reason.','commands':'After viewing fresh RGB images, write command.json atomically with exactly command_id (consecutive int from1), observed_index, skill, member, reason. One command at a time when phase idle/done. member null for position/release/finish/stop, or0south,1west,2north,3east for nail. Commands >30 observations old are rejected.','skills':{'position':'Two articulated parallel grippers approach opposite sheet edges, clamp, visually verify lift, withdraw physical shelves, align and lower. It DOES NOT RELEASE. Uses RGB XY/yaw and encoder-propagated height/tilt after grasp on a known level rack.','release':'Open grippers and withdraw sideways. Inspect placement before commanding.','nail':'After release, lower overhead carriage, visit selected member with the tool, record one pose-gated abstract nail, retract. One per member.','finish':'Declare visually finished; ends run.','stop':'End on visible failure or skill failure.'},'limitations':'Fast skills are handwritten, not Astra motor weights. Sheet height/tilt propagation assumes retained grasp; hidden slip is not fully observable. No physical nail or bond strength. Do not infer structural integrity. Prior frame assembly is a separate recorded run; station transfer is not simulated.','panel_size_m':PANEL_SIZE.tolist(),'nominal_panel_center_m':PANEL_GOAL.tolist()})
 try:
  for step in range(round(a.seconds/.02)):
   r=robot()
   if step%5==0:
    frames,cameras=e.images(ren);visual=v.update(frames,cameras)
    im=frames[1].copy()
    if display_camera is not None:
     ren.update_scene(e.d,camera=display_camera);im=ren.render().copy()
     if step==0:io.imwrite(out/'initial_overview.png',im)
    im[:240,-320:]=cv2.resize(frames[0],(320,240));cv2.rectangle(im,(0,800),(1280,960),(15,15,15),-1)
    label='ACTUAL ASTRA' if a.live else 'FIXED CAMERA BASELINE'
    for k,line in enumerate([label+' | PANEL STATION | '+s.state+' | %.1fs'%e.d.time,f'Panel fasteners: {len(s.events)}/4 | abstract events, no bond strength','Starts from completed frame; station transfer is not simulated']):cv2.putText(im,line,(18,840+35*k),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    writer.append_data(im)
    if step%50==0:
     paths=[]
     for k,f in enumerate(frames):p=public/f'{idx:04d}_camera{k}.png';io.imwrite(p,f);paths.append(str(p))
     packet={'observation_index':idx,'sim_time_s':e.d.time,'phase':s.state,'last_command_id':lastid,'last_failure':s.failure,'robot':r,'camera_calibration':cameras,'rgb':paths,'abstract_panel_nails':s.events};save(public/f'{idx:04d}.json',packet);save(public/'latest.json',packet);idx+=1
   missing=missing+.02 if visual is None else 0.
   est=held.estimate(visual,r) if s.holding and held.reference is not None and s.state!='release' else visual
   if missing>1 and s.state in ('approach','insert','close','lift','align','lower'):raise RuntimeError('RGB_lost_for_1s')
   if a.live:
    f=public/'command.json'
    if f.exists():
     c=json.loads(f.read_text())
     if c.get('command_id',0)>lastid:
      if set(c)!={'command_id','observed_index','skill','member','reason'} or c['command_id']!=lastid+1:raise ValueError('Invalid command schema/sequence')
      if not 0<=idx-1-c['observed_index']<=30:raise ValueError('Stale observation')
      if s.state not in ('idle','done','failed'):raise ValueError('Concurrent action')
      lastid+=1;actions.append({'command':c,'received_sim_s':e.d.time,'received_wall_s':time.monotonic()-started});save(private/'received_commands.json',actions)
      if c['skill'] in ('finish','stop'):finished=c['skill']=='finish';break
      s.start(c['skill'],c['member'],r,est)
   elif s.state in ('idle','done'):
    if action<len(sequence):kind,member=sequence[action];s.start(kind,member,r,est);actions.append({'skill':kind,'member':member,'time':e.d.time});action+=1
    else:
     settled+=.02
     if settled>3:finished=True;break
   cmd=s.command(r,est,visual)
   if s.begin_grasp:held.begin(est,r);s.begin_grasp=False
   e.rack_command=s.rack;e.overhead_command=s.overhead;e.step(*cmd);m=e.panel_metrics();trace.append({'time':e.d.time,'phase':s.state,'robot':clean(robot()),'estimate':clean(est),'visual':clean(visual),'metrics':m});raw.append(np.r_[e.d.time,e.d.qpos,e.d.qvel,e.d.ctrl,e.d.actuator_force])
   if s.state!=lastphase:print(round(e.d.time,2),s.state,flush=True);lastphase=s.state
   if not m['frame']['finite'] or any(m['frame']['warnings']) or e.peak>.0005:raise RuntimeError('Numerical acceptance failure:'+str(e.worst))
   if a.live:
    delay=started+(step+1)*.02-time.monotonic()
    if delay>0:time.sleep(delay)
    else:maxlag=max(maxlag,-delay)
  else:error='deadline'
 except Exception as exc:error=repr(exc);s.failure=error
 finally:
  frames,_=e.images(ren)
  for k,f in enumerate(frames):io.imwrite(out/f'final_camera{k}.png',f)
  if display_camera is not None:
   ren.update_scene(e.d,camera=display_camera);io.imwrite(out/'final_overview.png',ren.render().copy())
  writer.close();ren.close();save(private/'trace.json',trace);np.savez_compressed(private/'raw.npz',rows=raw,nq=e.m.nq,nv=e.m.nv,nu=e.m.nu);(private/'scene.xml').write_text(e.xml);m=e.panel_metrics();yaw=abs(np.arctan2(m['rotation_evaluator_only'][1][0],m['rotation_evaluator_only'][0][0]));success=finished and error is None and len(s.events)==4 and m['xy_error_m']<.006 and abs(m['height_error_m'])<.003 and m['tilt_deg']<1 and yaw<np.deg2rad(1) and max(m['forces']['grip'])<.1 and m['forces']['frame']>50 and m['frame']['frame_geometry_ok'] and e.peak<.0005
  save(out/'result.json',{'success':bool(success),'actual_astra':a.live,'model':'gpt-6-astra' if a.live else None,'privileged_diagnostic':False,'error':error,'finished':finished,'metrics':m,'abstract_panel_nails':s.events,'actions':actions,'simulated_s':e.d.time,'wall_s':time.monotonic()-started,'max_wall_lag_s':maxlag,'physics_paused_for_model':False,'seed':a.seed,'dt':a.dt,'physical_validation':False,'continuous_frame_to_panel_run':False,'embodiment':getattr(e,'embodiment','Separate fixed robot mounts'),'no_runpod':True});save(public/'trial_closed.json',{'reason':'finished' if finished else error,'evaluation':'private'})
if __name__=='__main__':main()
