"""Reproducible scripted camera baseline / clearly labeled privileged diagnostic."""
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,cv2
from .environment import Env,GOALS,ASSIGN,YAWS
from .skills import Skills
from .perception import Vision
from .plan_tool import Planner
from experiments.arm_corner.run import clean

def save(p,x):p.write_text(json.dumps(clean(x),indent=2))
def run(out,seed=0,condition='camera',dt=.001,limit=320,tool_only=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);e=Env(seed,dt);s=Skills(e.robot(),planner=Planner(e));v=Vision();r=mujoco.Renderer(e.m,960,1280);writer=io.get_writer(out/'trial.mp4',fps=10,codec='libx264',quality=7)
 if tool_only:
  # Explicit diagnostic initial condition: preassembled loose frame, no bonds.
  for i in range(4):
   qa=int(e.m.joint(f'wood{i}_free').qposadr[0]);e.d.qpos[qa:qa+3]=GOALS[i];e.d.qpos[qa+3:qa+7]=[np.cos(YAWS[i]/2),0,0,np.sin(YAWS[i]/2)]
  mujoco.mj_forward(e.m,e.d)
 actions=[{'skill':kind,'piece':p} for p in range(4) for kind in ('position','release')]+[{'skill':'nail','corner':c} for c in range(4)]
 if tool_only:actions=actions[8:]
 trace=[];raw=[];ix=0;est=[None]*4;started=time.monotonic();error=None;lastphase='';finalwait=0.
 try:
  for step in range(round(limit/.02)):
   robot=e.robot()
   if step%5==0:
    frames,cameras=e.images(r);est=v.update(frames,cameras,robot,s.holding)
    if condition=='oracle':
     est=[{'center':e.d.xpos[b].copy(),'yaw':float(np.arctan2(-e.d.xmat[b].reshape(3,3)[0,1],e.d.xmat[b].reshape(3,3)[1,1])),'time':e.d.time} for b in e.woods]
    display=frames[1].copy();small=cv2.resize(frames[0],(426,320));display[:320,-426:]=small
    cv2.putText(display,condition.upper()+' SCRIPTED BASELINE | '+s.state+' | %.1fs'%e.d.time,(10,355),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),2)
    cv2.putText(display,f'Abstract nails recorded: {len(s.events)}/4',(10,385),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),2);writer.append_data(display)
    if step%50==0:
     save(out/'progress.json',{'time':e.d.time,'action':ix,'phase':s.state,'failure':s.failure,'events':s.events,'metrics':e.metrics(),'est':est,'vision_diagnostics':v.diagnostics});io.imwrite(out/'latest.png',display)
   if s.state=='failed':error=s.failure;break
   if s.state in ('idle','done'):
    if ix<len(actions):s.start(actions[ix],est,robot);ix+=1
    else:
     finalwait+=.02
     if finalwait>3:break
   cmd=s.command(robot,est) if s.state not in ('idle','done','failed') else (s.targets,s.rotations,s.gripper)
   ik=e.step(*cmd);m=e.metrics();trace.append({'time':e.d.time,'phase':s.state,'action_index':ix,'robot':clean(robot),'vision':clean(est),'metrics':m,'ik':ik,'contacts':e.contacts if step%5==0 else None});raw.append(np.r_[e.d.time,e.d.qpos,e.d.qvel,e.d.ctrl,e.d.actuator_force])
   if s.state!=lastphase:print(round(e.d.time,2),ix,s.state,s.failure,flush=True);lastphase=s.state
   if not m['finite'] or any(m['warnings']) or e.peak>.0005:raise RuntimeError('Numerical acceptance failure: '+str(m['worst']))
  else:error='deadline'
 except Exception as exc:error=repr(exc)
 finally:
  frames,cameras=e.images(r)
  for k,f in enumerate(frames):io.imwrite(out/f'final_camera{k}.png',f)
  writer.close();r.close();save(out/'trace.json',trace);np.savez_compressed(out/'raw.npz',rows=raw,nq=e.m.nq,nv=e.m.nv,nu=e.m.nu);(out/'scene.xml').write_text(e.xml);m=e.metrics();save(out/'result.json',{'condition':condition,'tool_only_diagnostic':tool_only,'actual_astra':False,'error':error,'finished_actions':ix,'abstract_nails':s.events,'metrics':m,'parking_planning_wall_s':s.planning_times,'wall_s':time.monotonic()-started,'simulated_s':e.d.time,'success':bool(error is None and m['frame_geometry_ok'] and len(s.events)==4 and e.peak<.0005),'physical_validation':False})
  print(json.dumps(clean({'error':error,'metrics':m,'events':s.events})),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--condition',choices=['camera','oracle'],default='camera');ap.add_argument('--seed',type=int,default=0);ap.add_argument('--dt',type=float,default=.001);ap.add_argument('--tool-only',action='store_true');a=ap.parse_args();run(a.out,a.seed,a.condition,a.dt,tool_only=a.tool_only)
