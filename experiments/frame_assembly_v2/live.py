"""External live Astra camera decisions; physics runs during model reasoning."""
import argparse,json,time,hashlib,shutil
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,cv2
from .environment import Env,GOALS,SIZES
from .skills import Skills
from .perception import Vision
from .plan_tool import Planner
from experiments.arm_corner.run import clean

def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(clean(x),indent=2));q.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--seconds',type=float,default=900);ap.add_argument('--seed',type=int,default=0);a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);public=out/'controller';public.mkdir();private=out/'evaluator';private.mkdir();source=private/'source';source.mkdir()
 for folder in ('experiments/frame_assembly_v2','experiments/arm_corner'):
  dest=source/Path(folder).name;shutil.copytree(folder,dest,ignore=shutil.ignore_patterns('__pycache__'))
 save(private/'hashes.json',{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()})
 e=Env(a.seed);s=Skills(e.robot(),planner=Planner(e));v=Vision();ren=mujoco.Renderer(e.m,960,1280);writer=io.get_writer(out/'astra_frame.mp4',fps=10,codec='libx264',quality=8)
 save(public/'instructions.json',{'task':'Assemble four wood members into a flat square frame, then fire one ABSTRACT nail per corner with the articulated nailgun. No colored markers. Wood0 is south/bottom horizontal, wood1 west/left vertical, wood2 north/top horizontal, wood3 east/right vertical in overhead view. Handling arm0 southwest base handles wood0 and wood1; arm1 northeast handles wood2 and wood3. Arm2 inside the frame is the nailer. The frame is roughly 3 feet square. These are flat lumber members, not an upright box.',
 'allowed_directory':'ONLY this controller directory and referenced RGB images. Do not read source, evaluator, other runs, or hidden object states.',
 'commands':'Write command.json atomically with exactly command_id (consecutive int starting1), observed_index, skill, piece, corner, reason. piece is0..3 for pick/place/release and null otherwise. corner is0..3 for nail and null otherwise. Observe latest.json and the current RGB images before EACH separate command. Wait for phase idle/done/failed. Do not pre-submit or generate an entire fixed action sequence. Commands older than30 observations are rejected.',
 'control_update':'The fast controller now completes pickup through placement in one position command. Model chooses the member, inspects it before release, then chooses the next member or tool action. This avoids a long midair hold during model latency; it is not a stronger grasp model.',
 'skills':{'position':'piece0..3: one bounded composite skill approaches, grasps, visually verifies pickup, lifts, aligns with the known blueprint using RGB feedback, lowers and verifies convergence. It DOES NOT RELEASE. Use this atomic pickup-to-placement objective rather than pausing midair for another model call. On failure, stop.','release':'piece0..3: opens and retracts to park; rejected if visually outside placement tolerance','nail':'corner0SW,1SE,2NE,3NW: requires four released visually aligned members; arm2 approaches the corner, points diagonally across the butt joint, seats near wood, records one abstract firing event, retracts; duplicate nails rejected','finish':'declare visually complete and end trial','stop':'end trial with current poses held'},
 'blueprint_centers_m':GOALS.tolist(),'member_dimensions_m':SIZES.tolist(),
 'limits':'Skills are hand-written fast RGB/encoder feedback, not learned Astra motor weights. You choose tasks, order, visual acceptance/recovery. Primitive done is not evaluator success. Watch gaps, overlaps, dropped members. If a skill fails or visible geometry is wrong, stop and explain. Nails are event records only: no simulated nail body, bond strength, recoil, snapping or weld. This task tests sequencing/tool positioning, not structural integrity.',
 'sensors':'Calibrated overhead and two oblique RGB views, robot joint encoders/commands; no runtime object ground truth. Known geometry and initial feeder regions are priors. Physics runs while you reason. Model timing includes scheduling overhead.'})
 trace=[];raw=[];actions=[];est=[None]*4;idx=0;lastid=0;started=time.monotonic();error=None;finished=False;status='waiting';maxlag=0.;stable=0.
 try:
  for step in range(round(a.seconds/.02)):
   robot=e.robot()
   if step%5==0:
    frames,cameras=e.images(ren);est=v.update(frames,cameras,robot,s.holding);display=frames[1].copy();display[:320,-426:]=cv2.resize(frames[0],(426,320));cv2.putText(display,'ACTUAL ASTRA CAMERA TASK CONTROL | '+s.state+' | %.1fs'%e.d.time,(10,355),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),2);cv2.putText(display,f'Abstract nails recorded: {len(s.events)}/4 (no bond physics)',(10,385),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),2);writer.append_data(display)
    if step%50==0:
     paths=[]
     for k,f in enumerate(frames):
      p=public/f'{idx:04d}_camera{k}.png';io.imwrite(p,f);paths.append(str(p))
     packet={'observation_index':idx,'sim_time_s':e.d.time,'phase':s.state,'last_command_id':lastid,'last_primitive_failure':s.failure,'robot':robot,'camera_calibration':cameras,'rgb':paths,'command_status':status,'abstract_nails_recorded':s.events};save(public/f'{idx:04d}.json',packet);save(public/'latest.json',packet);idx+=1
   f=public/'command.json'
   if f.exists():
    command=json.loads(f.read_text())
    if command.get('command_id',0)>lastid:
     if set(command)!={'command_id','observed_index','skill','piece','corner','reason'}:raise ValueError('Invalid command schema')
     if command['command_id']!=lastid+1:raise ValueError('Nonconsecutive command')
     if not 0<=idx-1-command['observed_index']<=30:raise ValueError('Stale observation')
     if s.state not in ('idle','done','failed'):raise ValueError('Concurrent request')
     lastid+=1;record={'command':command,'received_wall_s':time.monotonic()-started,'received_sim_s':e.d.time};actions.append(record);save(private/'received_commands.json',actions)
     if command['skill'] in ('finish','stop'):finished=command['skill']=='finish';status=command['skill'];break
     s.start(command,est,robot);status='executing'
   cmd=s.command(robot,est) if s.state not in ('idle','done','failed') else (s.targets.copy(),s.rotations.copy(),s.gripper.copy());ik=e.step(*cmd);m=e.metrics();stable=stable+.02 if m['frame_geometry_ok'] and max(m['grip_force'])<.01 else 0.
   trace.append({'time':e.d.time,'phase':s.state,'robot':clean(robot),'vision':clean(est),'metrics':m,'ik':ik,'contacts':e.contacts if step%5==0 else None});raw.append(np.r_[e.d.time,e.d.qpos,e.d.qvel,e.d.ctrl,e.d.actuator_force])
   if not m['finite'] or any(m['warnings']) or e.peak>.0005:raise RuntimeError('Numerical acceptance failure '+str(m['worst']))
   delay=started+(step+1)*.02-time.monotonic()
   if delay>0:time.sleep(delay)
   else:maxlag=max(maxlag,-delay)
  else:error='deadline'
 except Exception as exc:error=repr(exc)
 finally:
  frames,cameras=e.images(ren)
  for k,f in enumerate(frames):io.imwrite(out/f'final_camera{k}.png',f)
  writer.close();ren.close();save(private/'trace.json',trace);np.savez_compressed(private/'raw.npz',rows=raw,nq=e.m.nq,nv=e.m.nv,nu=e.m.nu);(private/'scene.xml').write_text(e.xml);m=e.metrics()
  save(out/'result.json',{'actual_astra':True,'model':'gpt-6-astra','error':error,'primitive_failure':s.failure,'finished_by_astra':finished,'stable_released_s':stable,'abstract_nails':s.events,'strict_success':bool(finished and len(s.events)==4 and stable>=2 and m['frame_geometry_ok'] and e.peak<.0005),'metrics':m,'actions':actions,'parking_planning_wall_s':s.planning_times,'wall_s':time.monotonic()-started,'simulated_s':e.d.time,'max_wall_lag_s':maxlag,'physics_paused_for_model':False,'physical_validation':False,'no_runpod':True});save(public/'trial_closed.json',{'reason':'finished' if finished else error or status,'evaluator_results':'not exposed'})
if __name__=='__main__':main()
