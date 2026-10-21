"""Privileged-state feasibility diagnostic. Not Astra/camera policy evidence."""
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,cv2
from .environment import Env,PANEL_START,PANEL_GOAL,OFFSETS,SIDE_R,HOME
from experiments.arm_corner.run import clean

def run(out,seconds=100):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);e=Env();ren=mujoco.Renderer(e.m,720,960);video=io.get_writer(out/'trial.mp4',fps=10,codec='libx264',quality=7);t=e.targets.copy();R=e.rotations.copy();g=np.zeros(2);phase='approach';elapsed=0.;trace=[];err=None;begin=time.monotonic();last='';anchor=None
 try:
  for step in range(int(seconds/.02)):
   robot=e.robot();tcp=robot['tcp'];p=e.d.xpos[e.panel].copy();elapsed+=.02
   def switch(s):
    nonlocal phase,elapsed
    phase=s;elapsed=0.
   if phase!=last:print(round(e.d.time,2),phase,flush=True);last=phase
   if phase=='approach':
    t[:2]=p+OFFSETS; t[0,1]-=.12;t[1,1]+=.12
    if max(np.linalg.norm(t[:2]-tcp[:2],axis=1))<.004:switch('insert')
   elif phase=='insert':
    t[:2]=p+OFFSETS
    if max(np.linalg.norm(t[:2]-tcp[:2],axis=1))<.003:switch('close')
   elif phase=='close':
    g[:]=255
    if elapsed>2:anchor=tcp[:2].copy();switch('lift')
   elif phase=='lift':
    t[:2]=anchor+[0,0,.055]
    if p[2]>PANEL_START[2]+.045:anchor=tcp[:2].copy();switch('withdraw_shelves')
   elif phase=='withdraw_shelves':
    e.rack_command=.20
    if elapsed>3:switch('align')
   elif phase=='align':
    rp=e.d.xmat[e.panel].reshape(3,3); yaw=np.arctan2(rp[1,0],rp[0,0]); rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    t[:2]=np.array([0,0,PANEL_START[2]+.055])+(rp.T@(tcp[:2]-p).T).T;R[:2]=np.array([rp.T@r for r in robot['rotation'][:2]])
    if np.linalg.norm(p[:2])<.003 and abs(yaw)<.005:switch('lower')
   elif phase=='lower':
    rp=e.d.xmat[e.panel].reshape(3,3);yaw=np.arctan2(rp[1,0],rp[0,0]);rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    delta=np.array([0,0,PANEL_GOAL[2]+.0015])-p;delta[2]=np.clip(delta[2],-.0002,.0002);t[:2]=p+(rp.T@(tcp[:2]-p).T).T+delta;R[:2]=np.array([rp.T@r for r in robot['rotation'][:2]])
    if abs(p[2]-PANEL_GOAL[2]-.0015)<.001 and np.linalg.norm(p[:2])<.004 and e.panel_metrics()['tilt_deg']<.2:switch('release')
   elif phase=='release':
    g[:]=0
    if elapsed>2:switch('retreat')
   elif phase=='retreat':
    t[0,1]-=.0007;t[1,1]+=.0007
    if elapsed>4:switch('done')
   elif phase=='done':
    if elapsed>3:break
   if elapsed>35:raise RuntimeError('phase_timeout:'+phase)
   e.step(t,R,g);m=e.panel_metrics();trace.append({'time':e.d.time,'phase':phase,'robot':clean(e.robot()),'metrics':m})
   if step%5==0:
    ren.update_scene(e.d,camera='camera1');im=ren.render().copy();cv2.putText(im,'PRIVILEGED DIAGNOSTIC | '+phase,(12,35),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);video.append_data(im)
   if not m['frame']['finite'] or any(m['frame']['warnings']) or e.peak>.0005:raise RuntimeError('contact/numerical violation:'+str(e.worst))
 except Exception as x:err=repr(x)
 finally:
  frames,_=e.images(ren)
  for i,im in enumerate(frames):io.imwrite(out/f'final{i}.png',im)
  video.close();ren.close();(out/'scene.xml').write_text(e.xml);(out/'trace.json').write_text(json.dumps(clean(trace)));r={'privileged_diagnostic':True,'error':err,'phase':phase,'metrics':e.panel_metrics(),'simulated_s':e.d.time,'wall_s':time.monotonic()-begin};(out/'result.json').write_text(json.dumps(clean(r),indent=2));print(json.dumps(clean(r),indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();run(a.out)
