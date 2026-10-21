"""Actual camera-driven Astra task sequencer; wall-clock physics never waits on reasoning."""
import argparse,json,time,hashlib,shutil
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,cv2
from .environment import Env
from .skills import Skills
from .perception import Vision
from .run import clean,corner_metrics

def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(clean(x),indent=2));q.replace(p)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--seconds',type=float,default=600);a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);public=out/'controller';public.mkdir();private=out/'evaluator';private.mkdir();source=private/'source';source.mkdir()
 for p in Path('experiments/arm_corner').glob('*.py'):shutil.copy2(p,source/p.name)
 save(private/'hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')})
 e=Env(seed=1);s=Skills(e.robot());vision=Vision();ren=mujoco.Renderer(e.m,960,1280);writer=io.get_writer(out/'astra_corner.mp4',fps=10,codec='libx264',quality=8)
 save(public/'instructions.json',{'task':'Use camera images to assemble an unfastened L corner: the initially vertical-in-overhead-view member is wood0 and is handled by arm0 (right/front base). The initially horizontal member is wood1, handled by arm1 (rear/left base). Join wood0 upper end to the near face of wood1 at its right end, with outer right faces flush. Both must ultimately be released on the bench. Third arm/nailer is parked. No markers on wood.',
 'commands':'Write command.json atomically with exactly command_id (consecutive integer starting1), observed_index, skill, arm, reason. Read latest.json and view current images before each decision. Wait for phase done, idle, or failed before next command. You choose the sequence from camera observations. Do not submit an entire fixed sequence or copy old commands.',
 'skills':{'stabilize':'arm1: approach/grip reference member while table supports it','pick':'arm0: image-based approach/grasp, visually checked probe and lift','align_corner':'arm0: image-based alignment to wood1 right end, then lower to its height. Requires arm0 already holding. Does not release.','release':'arm0 or arm1: open jaws and retreat','stop':'stop task while holding current commanded pose','finish':'declare visual task complete, end recording; no evaluator success is revealed'},
 'limits':'Skills are hand-written fast feedback using calibrated RGB, known rectangular dimensions and robot encoders. They are not Astra motor weights. Joint limits/force limits/collisions remain active. A primitive reporting done is not proof of a correct joint: inspect cameras. If alignment is wrong or a skill fails, stop and explain, rather than silently claiming success.',
 'permitted':'ONLY this controller directory and referenced public images. Do not read source, evaluator, prior runs, latent object states or other files. You may use camera calibration and robot proprioception to reason. No hidden object pose or true joint condition is provided.',
 'nominal_member_size_m':[.038,.9144,.064], 'timing':'Physics continues while you reason; bounded joint feedback holds. Overall600-second local trial. Commands more than30 snapshots old are rejected. Sensor/model timing includes scheduling overhead; this is not hard real time.'})
 trace=[];raw=[];lastid=0;idx=0;est=[None,None];actions=[];heldtime=0.;pickup=False;stable=0.;maxlag=0.;late=0;started=time.monotonic();error=None;status='awaiting_first_command';finished=False
 try:
  for step in range(round(a.seconds/.02)):
   robot=e.robot()
   if step%5==0:
    frames,cameras=e.images(ren);est=vision.update(frames,cameras,robot,s.holding);display=np.concatenate(list(frames),axis=1);cv2.putText(display,'ACTUAL ASTRA TASK CONTROL | '+s.state+' | %.1fs'%e.d.time,(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(245,245,245),2);writer.append_data(display)
    if step%50==0:
     paths=[]
     for k,f in enumerate(frames):
      p=public/f'{idx:04d}_camera{k}.png';io.imwrite(p,f);paths.append(str(p))
     packet={'observation_index':idx,'sim_time_s':float(e.d.time),'phase':s.state,'last_command_id':lastid,'last_primitive_failure':s.failure,'robot':robot,'camera_calibration':cameras,'rgb':paths,'command_status':status};save(public/f'{idx:04d}.json',packet);save(public/'latest.json',packet);idx+=1
   commandfile=public/'command.json'
   if commandfile.exists():
    command=json.loads(commandfile.read_text())
    if command.get('command_id',0)>lastid:
     if set(command)!={'command_id','observed_index','skill','arm','reason'}:raise ValueError('Invalid command schema')
     if command['command_id']!=lastid+1:raise ValueError('Nonconsecutive command')
     if not 0<=idx-1-command['observed_index']<=30:raise ValueError('Stale/nonexistent observation')
     if s.state not in ('idle','done','failed'):raise ValueError('Concurrent primitive request')
     if command['arm'] not in (0,1):raise ValueError('Only handling arms allowed')
     lastid+=1;record={'command':command,'received_wall_s':time.monotonic()-started,'received_sim_s':float(e.d.time)};actions.append(record);save(private/'received_commands.json',actions)
     if command['skill'] in ('finish','stop'):finished=command['skill']=='finish';status=command['skill'];break
     s.start(command,est,robot);status='executing'
   if s.state not in ('idle','done','failed'):cmd=s.command(robot,est)
   else:cmd=(s.targets.copy(),s.rotations.copy(),s.gripper.copy())
   ik=e.step(*cmd);m=corner_metrics(e);heldtime=heldtime+.02 if e.d.xpos[e.woods[0],2]>.09 and m['grip_force'][0]>1 and m['table_support'][0]<.01 else 0.;pickup|=heldtime>=1;stable=stable+.02 if m['strict_geometry'] and m['valid'] and max(m['grip_force'])<.01 and m['moving_speed']<.005 else 0.
   trace.append({'time':float(e.d.time),'phase':s.state,'robot':clean(robot),'vision':clean(est),'metrics':m,'ik':ik,'contacts':e.contacts});raw.append(np.r_[e.d.time,e.d.qpos,e.d.qvel,e.d.ctrl,e.d.actuator_force])
   if not m['valid']:raise RuntimeError('Numerical acceptance failure')
   target=started+(step+1)*.02;delay=target-time.monotonic()
   if delay>0:time.sleep(delay)
   else:late+=1;maxlag=max(maxlag,-delay)
  else:error='overall_deadline'
 except Exception as exc:error=repr(exc)
 finally:
  writer.close();ren.close();m=corner_metrics(e);save(private/'trace.json',trace);np.savez_compressed(private/'raw.npz',rows=raw,nq=e.m.nq,nv=e.m.nv,nu=e.m.nu);(private/'scene.xml').write_text(e.xml)
  save(out/'result.json',{'actual_astra':True,'model':'gpt-6-astra','error':error,'primitive_failure':s.failure,'finished_by_astra':finished,'pickup':pickup,'stable_released_s':stable,'strict_success':bool(finished and pickup and stable>=2 and m['valid']),'metrics':m,'actions':actions,'wall_s':time.monotonic()-started,'simulated_s':float(e.d.time),'max_wall_lag_s':maxlag,'late_cycles':late,'physics_paused_for_model':False,'physical_validation':False,'no_runpod':True});save(public/'trial_closed.json',{'reason':'finished' if finished else error or status,'evaluator_results':'not exposed'})
if __name__=='__main__':main()
