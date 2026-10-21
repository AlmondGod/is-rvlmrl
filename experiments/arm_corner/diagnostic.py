import argparse,json,time
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,cv2
from .environment import Env
from .skills import Skills

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--dt',type=float,default=.001);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);e=Env(dt=a.dt);s=Skills(e.robot());ren=mujoco.Renderer(e.m,720,960);v=io.get_writer(out/'video.mp4',fps=10);trace=[];raw=[];actions=[{'skill':'pick','arm':0},{'skill':'align_corner','arm':0},{'skill':'release','arm':0}];start=time.monotonic();last=''
 def estimate():
  return [{'center':e.d.xpos[b].copy(),'yaw':float(np.arctan2(-e.d.xmat[b].reshape(3,3)[0,1],e.d.xmat[b].reshape(3,3)[1,1])),'time':float(e.d.time)} for b in e.woods]
 for step in range(3000):
  est=estimate()
  if s.state in ('idle','done'):
   if not actions:break
   s.start(actions.pop(0),est,e.robot())
  cmd=s.command(e.robot(),est);errors=e.step(*cmd);m=e.metrics();trace.append({'time':float(e.d.time),'state':s.state,'metrics':m,'ik':errors,'targets':s.targets.tolist()});raw.append(np.r_[e.d.time,e.d.qpos,e.d.qvel,e.d.ctrl,e.d.actuator_force])
  if step%5==0:
   frames,c=e.images(ren);f=np.concatenate(list(frames),axis=1);cv2.putText(f,'PRIVILEGED DIAGNOSTIC | '+s.state,(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(240,240,240),2);v.append_data(f)
   if step==0:io.imwrite(out/'initial.png',f)
  if s.state!=last:print(s.state,round(float(e.d.time),2),'tcp',e.robot()['tcp'][0],'object',est[0]['center'],'peak',e.peak,flush=True);last=s.state
  if s.state=='failed' or e.peak>.005 or not m['finite']:break
 v.close();ren.close();(out/'result.json').write_text(json.dumps({'condition':'privileged_diagnostic','metrics':m,'failure':s.failure,'phase':s.state,'sim_time':float(e.d.time),'wall_time':time.monotonic()-start,'ik_initial':e.ikerrors},indent=2));(out/'trace.json').write_text(json.dumps(trace));np.savez_compressed(out/'raw.npz',rows=raw,nq=e.m.nq,nv=e.m.nv,nu=e.m.nu);(out/'scene.xml').write_text(e.xml);print((out/'result.json').read_text())
if __name__=='__main__':main()
