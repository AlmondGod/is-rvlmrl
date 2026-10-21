"""Separate diagnostic/camera baseline runner; the scripted sequence is NOT Astra."""
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,cv2
from .environment import Env,SIZE
from .skills import Skills
from .perception import Vision

def clean(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 return x

def corner_metrics(e):
 d=e.d;w=e.woods;p,q=[d.xpos[b] for b in w];R,S=[d.xmat[b].reshape(3,3) for b in w];u=R[:,1];v=S[:,0];ref=-S[:,1]
 gap=float((q-v*SIZE[0]/2-p-u*SIZE[1]/2)@v);flush=float((p+R[:,0]*SIZE[0]/2-q-ref*SIZE[1]/2)@ref);height=float(abs(p[2]-q[2]));base=e.metrics();base.update(gap_m=gap,outer_flush_m=flush,height_error_m=height)
 qa=int(e.m.joint('wood0_free').dofadr[0]);base['moving_speed']=float(np.linalg.norm(d.qvel[qa:qa+3]));base['strict_geometry']=0<=gap<=.005 and abs(flush)<.005 and height<.003 and base['square_error_deg']<2
 base['valid']=base['finite'] and not any(base['warnings']) and e.peak<.0005
 return base

def run(out,seed=0,condition='camera',dt=.001):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);e=Env(seed,dt);s=Skills(e.robot());vision=Vision();ren=mujoco.Renderer(e.m,960,1280);video=io.get_writer(out/'video.mp4',fps=10,codec='libx264',quality=8);trace=[];raw=[];est=[None,None];images=[];actions=[{'skill':'stabilize','arm':1},{'skill':'pick','arm':0},{'skill':'align_corner','arm':0},{'skill':'release','arm':0},{'skill':'release','arm':1}];start=time.monotonic();last='';heldtime=0.;pickup=False;stable=0.;poseerrors=[];final_wait=0
 try:
  for step in range(3500):
   robot=e.robot()
   if step%5==0:
    frames,c=e.images(ren);est=vision.update(frames,c,robot,s.holding);images.append(frames[0].copy());f=np.concatenate(list(frames),axis=1);cv2.putText(f,condition.upper()+' BASELINE | '+s.state+' | %.1fs'%e.d.time,(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(240,240,240),2);video.append_data(f)
    if step==0:io.imwrite(out/'initial.png',f);(out/'calibration.json').write_text(json.dumps(c))
    for i,x in enumerate(est):
     if x is not None:poseerrors.append({'time':float(e.d.time),'wood':i,'error_m':float(np.linalg.norm(x['center']-e.d.xpos[e.woods[i]])),'estimate':clean(x)})
   if condition=='oracle':est=[{'center':e.d.xpos[b].copy(),'yaw':float(np.arctan2(-e.d.xmat[b].reshape(3,3)[0,1],e.d.xmat[b].reshape(3,3)[1,1])),'time':float(e.d.time)} for b in e.woods]
   if s.state in ('idle','done'):
    if not actions:
     final_wait+=1
     if final_wait>150:break
    else:s.start(actions.pop(0),est,robot)
   cmd=s.command(robot,est);err=e.step(*cmd);m=corner_metrics(e);heldtime=heldtime+.02 if e.d.xpos[e.woods[0],2]>.09 and m['grip_force'][0]>1 and m['table_support'][0]<.01 else 0.;pickup|=heldtime>=1;stable=stable+.02 if m['strict_geometry'] and m['valid'] and m['grip_force'][0]<.01 and m['moving_speed']<.005 else 0.
   trace.append({'time':float(e.d.time),'phase':s.state,'robot':clean(robot),'vision':clean(est),'metrics':m,'targets':clean(cmd),'ik_error':err,'contacts':e.contacts});raw.append(np.r_[e.d.time,e.d.qpos,e.d.qvel,e.d.ctrl,e.d.actuator_force])
   if s.state!=last:print(s.state,round(float(e.d.time),2),'gap',m['gap_m'],'flush',m['outer_flush_m'],'penetration',e.peak,flush=True);last=s.state
   if s.state=='failed' or e.peak>.005 or not m['finite']:break
 finally:video.close();ren.close()
 result={'condition':condition,'online_astra':False,'seed':seed,'dt':dt,'metrics':m,'failure':s.failure,'phase':s.state,'pickup':pickup,'released_stable_s':stable,'strict_success':bool(pickup and stable>=2 and m['valid']),'simulated_s':float(e.d.time),'wall_s':time.monotonic()-start,'perception_errors':poseerrors,'physical_validation':False,'no_runpod':True};(out/'result.json').write_text(json.dumps(result,indent=2));(out/'trace.json').write_text(json.dumps(trace));np.savez_compressed(out/'raw.npz',rows=raw,nq=e.m.nq,nv=e.m.nv,nu=e.m.nu);np.savez_compressed(out/'controller_rgb.npz',rgb=images);(out/'scene.xml').write_text(e.xml);print(json.dumps({k:result[k] for k in ('failure','phase','pickup','strict_success','metrics')}),flush=True);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--condition',default='camera',choices=['camera','oracle']);p.add_argument('--dt',type=float,default=.001);a=p.parse_args();run(a.out,a.seed,a.condition,a.dt)
if __name__=='__main__':main()
