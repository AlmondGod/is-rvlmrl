"""Privileged checkpoint initialization ONLY; RGB/encoder flip controller.
Reconstructed fasteners test mechanics, not continuous fastening/policy success.
"""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,cv2
from scipy.spatial.transform import Rotation
from .environment import Env
from .planner import Planner
from .fasteners import NailBank,NailParameters
from .flip import Flip
from .trace_io import iter_trace
from .flip_evaluator import reference,measure,passes
from experiments.wall_panel.perception import Vision
from experiments.arm_corner.run import clean

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);ap.add_argument('--condition',choices=['nails','unfastened','weak'],default='nails');ap.add_argument('--dt',type=float,default=.001);ap.add_argument('--contact-time',type=float,default=.004);ap.add_argument('--grasp',choices=['sheet','stack'],default='sheet');ap.add_argument('--smooth',action='store_true');ap.add_argument('--clearance',action='store_true');ap.add_argument('--outboard',action='store_true');ap.add_argument('--fixture-half-width',type=float,default=.46);ap.add_argument('--start-time',type=float);ap.add_argument('--shoulder-half-span',type=float,default=.34);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);shutil.copytree(Path(__file__).parent,out/'source',ignore=shutil.ignore_patterns('__pycache__'));shutil.copytree(Path('experiments/frame_assembly_v2'),out/'planner_dependency',ignore=shutil.ignore_patterns('__pycache__'));(out/'arguments.json').write_text(json.dumps(vars(a),indent=2));src=Path(a.source);raw=np.load(src/'raw.npz')['rows'];last_est=None
 for row in iter_trace(src/'trace.json'):
  if row.get('est') is not None:last_est=row['est']
 result=json.loads((src/'result.json').read_text());e=Env(dt=a.dt,fixture_half_width=a.fixture_half_width,shoulder_half_span=a.shoulder_half_span);m,d=e.m,e.d;initial_robot=[d.qpos[q].copy() for q in e.qids]
 for name in [f'wood{i}_geom' for i in range(4)]+['panel_geom']:m.geom_solref[m.geom(name).id,0]=a.contact_time
 def restore(row):
  d.qpos[:]=row[1:1+m.nq];d.qvel[:]=row[1+m.nq:1+m.nq+m.nv];d.ctrl[:]=row[1+m.nq+m.nv:1+m.nq+m.nv+m.nu];mujoco.mj_forward(m,d)
 histories=[raw];ancestor=result.get('diagnostic_checkpoint')
 while ancestor:
  ancestor_path=Path(ancestor);histories.append(np.load(ancestor_path/'raw.npz')['rows']);ancestor=json.loads((ancestor_path/'result.json').read_text()).get('diagnostic_checkpoint')
 params=NailParameters(withdrawal_N=1.,lateral_N=1.) if a.condition=='weak' else NailParameters();bank=NailBank(e,params,enabled=a.condition!='unfastened')
 # Initial-condition reconstruction only: replay tool/material configurations
 # at recorded shots, then initialize the final assembled configuration.
 for ev in result['events']:
  history=next((rows for rows in histories if rows[0,0]-.021<=ev['time']<=rows[-1,0]+.021),None)
  if history is None:raise ValueError('No recorded insertion state for '+str(ev))
  restore(history[np.argmin(abs(history[:,0]-ev['time']))]);bank.fire(ev['kind'],ev['index'])
 restore(raw[-1] if a.start_time is None else raw[np.argmin(abs(raw[:,0]-a.start_time))]);
 if a.shoulder_half_span!=result.get('shoulder_half_span_m',.34):
  # New hardware checkpoint starts with empty grippers parked, before time zero.
  for arm in (0,1):d.qpos[e.qids[arm]]=initial_robot[arm];d.qvel[e.vids[arm]]=0
  mujoco.mj_forward(m,d)
 d.time=0.;e.nails=bank;e._joint_internal_forces=bank.apply;e.shuttle_command=0.;e.rack_command=.2;e.targets=e.robot()['tcp'].copy();e.rotations=e.robot()['rotation'].copy();e.qtarget=[d.qpos[q].copy() for q in e.qids];e.ik_cache={};e.peak=0.;e.worst={}
 renderer=mujoco.Renderer(m,960,1280);frames,cams=e.images(renderer);panel=Vision().update(frames,cams)
 # Stored visual estimates are used only to initialize a planning obstacle map.
 # Current sheet target is remeasured from RGB. This is explicitly diagnostic.
 est=last_est
 for x in est:
  for k in ('center','rotation'):x[k]=np.array(x[k])
 pl=Planner(e);controller=Flip(e.robot(),pl,est,panel,grasp=a.grasp,smooth=a.smooth,clearance=a.clearance,outboard=a.outboard);ref=reference(e);writer=io.get_writer(out/'flip.mp4',fps=10,codec='libx264',quality=7);rows=[];log=[];error=None;wall=time.monotonic();initial=[d.xmat[e.panel].reshape(3,3).T@(d.xpos[b]-d.xpos[e.panel]) for b in e.woods];relR=[d.xmat[e.panel].reshape(3,3).T@d.xmat[b].reshape(3,3) for b in e.woods];last=''
 try:
  for step in range(7000):
   r=e.robot();r.update(shuttle_x=float(d.qpos[int(m.joint('shuttle_x').qposadr[0])]),rack_slide0=float(d.qpos[int(m.joint('rack_slide0').qposadr[0])]),rack_slide1=float(d.qpos[int(m.joint('rack_slide1').qposadr[0])]))
   cmd=controller.command(r);e.step(*cmd);R=d.xmat[e.panel].reshape(3,3);errors=[float(np.linalg.norm(R.T@(d.xpos[b]-d.xpos[e.panel])-p)) for b,p in zip(e.woods,initial)];angles=[float(Rotation.from_matrix((R.T@d.xmat[b].reshape(3,3))@p.T).magnitude()) for b,p in zip(e.woods,relR)]
   contacts=[]
   for ci,c in enumerate(d.contact):
    names=[m.geom(g).name for g in (c.geom1,c.geom2)]
    if any(n in ['panel_geom']+[f'wood{i}_geom' for i in range(4)] for n in names):
     f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);contacts.append(dict(pair=names,depth_m=float(-c.dist),force=f.tolist(),position=c.pos.tolist()))
   log.append(dict(contacts=contacts,retention=measure(e,ref),time=d.time,phase=controller.state,relative_position_errors_m=errors,relative_angle_errors_rad=angles,panel_up_dot=float(R[2,2]),fasteners=bank.metrics(),peak_penetration_m=e.peak));rows.append(np.r_[d.time,d.qpos,d.qvel,d.ctrl,d.actuator_force])
   if step%50==0:(out/'progress.json').write_text(json.dumps(clean(dict(time=d.time,phase=controller.state,tcp=e.robot()['tcp'],target=controller.targets,error_m=np.linalg.norm(controller.targets[:2]-e.robot()['tcp'][:2],axis=1),fasteners=bank.metrics(),peak_penetration_m=e.peak)),indent=2))
   if step%5==0:
    renderer.update_scene(d,camera='torso_overview');im=renderer.render().copy();cv2.rectangle(im,(0,815),(1280,960),(15,15,15),-1)
    for j,line in enumerate(['PRIVILEGED CHECKPOINT / RGB-ENCODER FLIP DIAGNOSTIC',f'{a.condition} | {controller.state} | {d.time:.1f}s | intact nails {bank.metrics()["intact"]}/8','Provisional finite-strength joints; reconstructed insertion; NOT Astra']):cv2.putText(im,line,(15,850+35*j),cv2.FONT_HERSHEY_SIMPLEX,.62,(245,245,245),2)
    writer.append_data(im)
   if controller.state!=last:print(round(d.time,2),controller.state,flush=True);last=controller.state
   if e.peak>.0005:raise RuntimeError('penetration gate '+str(e.worst))
   if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):raise RuntimeError('numerical failure')
   if controller.state=='done':break
  else:error='deadline'
 except Exception as ex:error=repr(ex);print(error,flush=True)
 finally:
  writer.close();renderer.close();(out/'scene.xml').write_text(e.xml);np.savez_compressed(out/'raw.npz',rows=rows);(out/'trace.json').write_text(json.dumps(clean(log)));last=log[-1] if log else {};success=bool(error is None and passes(measure(e,ref),bank.metrics()['intact'],controller.state=='done'))
  summary=dict(outboard=a.outboard,shoulder_half_span=a.shoulder_half_span,clearance=a.clearance,grasp=a.grasp,smooth=a.smooth,fixture_half_width=a.fixture_half_width,start_time=a.start_time,success=success,condition=a.condition,error=error,last=last,wall_s=time.monotonic()-wall,privileged_checkpoint_initialization=True,continuous_assembly=False,actual_astra=False,physical_validation=False,source=str(src),dt=a.dt,contact_time_s=a.contact_time);(out/'result.json').write_text(json.dumps(clean(summary),indent=2));print(json.dumps(clean(summary)),flush=True)
if __name__=='__main__':main()
