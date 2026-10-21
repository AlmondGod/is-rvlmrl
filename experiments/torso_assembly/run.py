"""One scene, one physics clock, no reset between loose members and wall.
Camera feedback baseline; this runner does not claim an Astra policy trial.
"""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,cv2
from experiments.arm_corner.run import clean
from .environment import Env,FIXTURE_HEIGHT,SIDE_R,OFFSETS
from .planner import Planner
from .frame_skills import Skills as FrameSkills
from .frame_vision import MultiVision as FrameVision,probe_anchor
from .panel_skills import Skills as PanelSkills
from .tool_skills import Tool
from experiments.wall_panel.perception import Vision as PanelVision,HeldSheet

def save(p,x):p.write_text(json.dumps(clean(x),indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--layout-seed',type=int);ap.add_argument('--physical-nails',action='store_true');ap.add_argument('--flip',action='store_true');ap.add_argument('--flip-profile',choices=['legacy','clearance-stack'],default='clearance-stack');ap.add_argument('--shoulder-half-span',type=float);ap.add_argument('--seconds',type=float,default=700);ap.add_argument('--dt',type=float);ap.add_argument('--oracle',action='store_true');ap.add_argument('--north-lift-height',type=float);ap.add_argument('--resume-diagnostic');ap.add_argument('--camera-after-resume',action='store_true');ap.add_argument('--resume-row',type=int,default=-1);ap.add_argument('--resume-stage',type=int,default=0);a=ap.parse_args()
 if a.dt is None:a.dt=.0005 if a.flip and a.flip_profile=='clearance-stack' else .001
 if a.flip and not a.physical_nails:raise ValueError('--flip requires --physical-nails')
 if a.shoulder_half_span is None:a.shoulder_half_span=.55 if a.flip and a.flip_profile=='clearance-stack' else .34
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);shutil.copytree('experiments/torso_assembly',out/'source',ignore=shutil.ignore_patterns('__pycache__'))
 for dependency in ('frame_assembly_v2','arm_corner','wall_panel','box_assembly','astra_camera'):shutil.copytree('experiments/'+dependency,out/'source_dependencies'/dependency,ignore=shutil.ignore_patterns('__pycache__'))
 save(out/'experiment.json',{'layout_seed_evaluator_only':a.layout_seed,'distribution':'face-up, surveyed support height; within-side lane permutation; XY jitter ±12/35 mm, yaw ±0.06 rad','privileged_diagnostic':a.oracle,'flip_profile':a.flip_profile,'shoulder_half_span_m':a.shoulder_half_span,'fixture_access_slots':a.flip and a.flip_profile=='clearance-stack'})
 e=Env(a.dt,layout_seed=a.layout_seed,physical_nails=a.physical_nails,fixture_half_width=.38 if a.flip and a.flip_profile=='clearance-stack' else .46,shoulder_half_span=a.shoulder_half_span);pl=Planner(e);fs=FrameSkills(e.robot(),pl);fv=FrameVision();pv=PanelVision();held=HeldSheet();ps=None;tool=None;ren=mujoco.Renderer(e.m,960,1280);writer=io.get_writer(out/'continuous.mp4',fps=10,codec='libx264',quality=7)
 def robot():
  r=e.robot()
  for name in ('shuttle_x','rack_slide0','rack_slide1'):r[name]=float(e.d.qpos[int(e.m.joint(name).qposadr[0])])
  return r
 stages=[('position',i) for i in range(4)];sequence=[(kind,i) for i in range(4) for kind in ('position','release')]+[('clear_arm',i) for i in range(2)]+[('frame_nail',i) for i in range(4)]+[('feed',None),('panel_prepare',0),('panel_prepare',1),('panel_position',None),('panel_release',None)]+[('panel_nail',i) for i in range(4)]
 if a.flip:sequence.append(('flip',None))
 flip=None;flip_reference=None;pre_flip=None
 idx=0;active=False;stage='idle';last='';trace=[];raw=[];events=[];est=[None]*4;visual=None;sheet=None;error=None;finished=False;timer=0.;start=time.monotonic();cmd=(e.targets,e.rotations,e.gripper,None);finalwait=0.
 if a.resume_diagnostic:
  if not (a.oracle or a.camera_after_resume):raise ValueError('Checkpoint initialization is privileged; choose --oracle or --camera-after-resume for the diagnostic controller')
  rows=np.load(Path(a.resume_diagnostic)/'raw.npz')['rows'];row=rows[a.resume_row]
  if a.camera_after_resume:
   from .trace_io import iter_trace
   for saved in iter_trace(Path(a.resume_diagnostic)/'trace.json'):
    if saved['time']>row[0]+1e-8:break
    if saved.get('est') is not None:est=saved['est']
   est=[dict(x,center=np.array(x['center'])) for x in est]
   previous=json.loads((Path(a.resume_diagnostic)/'result.json').read_text())
   events=[ev for ev in previous['events'] if ev['time']<=row[0]]
   if e.nails is not None:
    for ev in events:
     shot=rows[np.argmin(abs(rows[:,0]-ev['time']))];e.d.qpos[:]=shot[1:1+e.m.nq];mujoco.mj_forward(e.m,e.d);e.nails.fire(ev['kind'],ev['index'])
  if len(row)!=1+e.m.nq+e.m.nv+2*e.m.nu:raise ValueError('Model state dimensions changed')
  gun_reset=(row[1:1+e.m.nq][e.qids[2]].copy() if a.camera_after_resume else e.d.qpos[e.qids[2]].copy());e.d.time=row[0];e.d.qpos[:]=row[1:1+e.m.nq];e.d.qpos[e.qids[2]]=gun_reset;e.d.qvel[:]=row[1+e.m.nq:1+e.m.nq+e.m.nv];e.d.qvel[e.vids[2]]=0;e.d.ctrl[:]=row[1+e.m.nq+e.m.nv:1+e.m.nq+e.m.nv+e.m.nu];mujoco.mj_forward(e.m,e.d)
  e.shuttle_command=float(e.d.qpos[int(e.m.joint('shuttle_x').qposadr[0])]);e.rack_command=float(e.d.qpos[int(e.m.joint('rack_slide0').qposadr[0])]);e.gripper=e.d.ctrl[e.grips].copy();e.qtarget=[e.d.qpos[q].copy() for q in e.qids];e.targets=e.robot()['tcp'].copy();e.rotations=e.robot()['rotation'].copy();e.ik_cache={};fs=FrameSkills(e.robot(),pl);idx=a.resume_stage;cmd=(e.targets,e.rotations,e.gripper,None)
 try:
  for step in range(round(a.seconds/.02)):
   r=robot()
   if step%5==0:
    if a.oracle:
     ren.update_scene(e.d,camera='camera0');frames=[ren.render().copy()];cameras=None
    else:frames,cameras=e.images(ren)
    if idx<=13 and not a.oracle:est=fv.update(frames,cameras,r,fs.holding,probe_anchor(fs))
    if idx>=13 and r['shuttle_x']<.03 and not a.oracle:visual=pv.update(frames,cameras)
    if a.oracle:
     est=[{'center':e.d.xpos[b].copy(),'yaw':float(np.arctan2(-e.d.xmat[b].reshape(3,3)[0,1],e.d.xmat[b].reshape(3,3)[1,1])),'time':e.d.time,'tcp_at_measurement':r['tcp'][0 if i<2 else 1].copy()} for i,b in enumerate(e.woods)]
     b=e.panel;R=e.d.xmat[b].reshape(3,3).copy();visual={'center':e.d.xpos[b].copy(),'rotation':R,'yaw':float(np.arctan2(R[1,0],R[0,0]))}
    ren.update_scene(e.d,camera='torso_overview');im=ren.render().copy();im[:240,-320:]=cv2.resize(frames[0],(320,240));cv2.rectangle(im,(0,810),(1280,960),(15,15,15),-1)
    for k,line in enumerate([('PRIVILEGED DIAGNOSTIC' if a.oracle else 'CAMERA FEEDBACK / CHECKPOINT DIAGNOSTIC' if a.resume_diagnostic else 'CAMERA FEEDBACK BASELINE')+' | '+stage+' | %.1f s'%e.d.time,('Privileged checkpoint initialization; no subsequent workpiece reset' if a.resume_diagnostic else 'Continuous physics; free workpieces; no scene reset'),f'{"Finite nail triggers" if a.physical_nails else "Abstract fasteners"} {len(events)}/8 | new torso + UR10e embodiment']):cv2.putText(im,line,(20,850+35*k),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    writer.append_data(im)
    if step%50==0:io.imwrite(out/'latest.png',im);save(out/'progress.json',{'time':e.d.time,'stage':stage,'index':idx,'est':est,'visual':visual,'metrics':e.panel_metrics()})
   sheet=held.estimate(visual,r) if ps and ps.holding and held.reference is not None and ps.state!='release' else visual
   if not active:
    if idx==len(sequence):
     finalwait+=.02
     if finalwait>3:finished=True;break
    else:
     kind,i=sequence[idx];idx+=1;timer=0.;active=True
     if kind in ('position','release'):
      fs.start({'skill':kind,'piece':i},est,r)
      if i==2 and a.north_lift_height is not None:fs.lift_height=a.north_lift_height
     elif kind.endswith('nail'):tool=Tool(r,pl,'frame' if kind=='frame_nail' else 'panel',i,est,sheet if kind=='panel_nail' else None)
     elif kind=='clear_arm':
      target=np.array([-.90 if i==0 else .90,.65,.65]);prepare_route=np.array(pl.plan(i,r,est,target,fs.rotations[i]));prepare_index=1
     elif kind=='panel_prepare':
      target=np.array(sheet['center'])+OFFSETS[i];target[2]=sheet['center'][2];target[0]+=(-.12 if i==0 else .12);prepare_route=np.array(pl.plan(i,r,est,target,SIDE_R[i],sheet));prepare_index=1
     elif kind=='panel_position':ps=PanelSkills(r,pl,est);ps.start('position',None,r,sheet)
     elif kind=='panel_release':ps.start('release',None,r,sheet)
     elif kind=='flip':
      from .flip import Flip
      flip=Flip(r,pl,est,visual,**(dict(grasp='stack',smooth=True,clearance=True) if a.flip_profile=='clearance-stack' else {}));pre_flip=e.panel_metrics();R=e.d.xmat[e.panel].reshape(3,3);flip_reference=[(R.T@(e.d.xpos[b]-e.d.xpos[e.panel]),R.T@e.d.xmat[b].reshape(3,3)) for b in e.woods]
   if active:
    timer+=.02
    if kind=='flip':
     cmd=flip.command(r);stage='flip: '+flip.state;active=flip.state!='done'
    elif kind in ('position','release'):
     cmd=fs.command(r,est);stage=f'frame {i}: {fs.state}'
     if fs.state=='failed':raise RuntimeError(fs.failure)
     active=fs.state!='done'
    elif kind.endswith('nail'):
     cmd=tool.command(r);stage=f'{kind} {i}: {tool.state}'
     if e.nails is not None and tool.events and not getattr(tool,'shot_applied',False):
      e.nails.fire(tool.kind,i);tool.shot_applied=True
      tool.events[0].update(abstract=False,physical_bond=None,triggered=True)
     if tool.state=='done':events+=tool.events;active=False;cmd=(r['tcp'].copy(),r['rotation'].copy(),np.zeros(2),None)
    elif kind=='feed':
     stage='physical plywood feed';e.shuttle_command=max(0.,e.shuttle_command-.0006);cmd=(r['tcp'].copy(),r['rotation'].copy(),np.zeros(2),{a:r['joints'][a] for a in range(3)})
     if r['shuttle_x']<.001 and timer>47:active=False
     if timer>65:raise RuntimeError('physical shuttle failed to reach target within 65 s')
    elif kind in ('panel_prepare','clear_arm'):
     stage=f'{kind} arm {i}';cmd=(r['tcp'].copy(),r['rotation'].copy(),np.zeros(2),{i:prepare_route[prepare_index]})
     if np.max(abs(r['joints'][i]-prepare_route[prepare_index]))<.01:
      prepare_index+=1
      if prepare_index>=len(prepare_route):active=False
     if timer>45:raise RuntimeError('panel approach timeout')
    else:
     cmd=ps.command(r,sheet,visual);stage='panel: '+ps.state;e.rack_command=ps.rack
     if ps.begin_grasp:held.begin(sheet,r);ps.begin_grasp=False
     active=ps.state!='done'
   if stage!=last:print(round(e.d.time,2),stage,flush=True);last=stage
   ik=e.step(*cmd);m=e.panel_metrics()
   if flip_reference is not None:
    from .flip_evaluator import measure
    m['flip']=measure(e,flip_reference)
   if e.nails is not None:m['fasteners']=e.nails.metrics()
   trace.append({'time':e.d.time,'stage':stage,'index':idx,'robot':clean(robot()),'est':clean(est),'vision_diagnostics':clean(fv.diagnostics),'sheet':clean(sheet),'metrics':m,'ik':ik});raw.append(np.r_[e.d.time,e.d.qpos,e.d.qvel,e.d.ctrl,e.d.actuator_force])
   if e.peak>.0005 or any(m['frame']['warnings']) or not m['frame']['finite']:raise RuntimeError('numerical gate: '+str(e.worst))
  else:error='deadline'
 except KeyboardInterrupt:error='development_interrupted';print(error,flush=True)
 except Exception as ex:error=repr(ex);print(error,flush=True)
 finally:
  ren.update_scene(e.d,camera='torso_overview');io.imwrite(out/'final.png',ren.render());ren.close();writer.close();save(out/'trace.json',trace);np.savez_compressed(out/'raw.npz',rows=raw,nq=e.m.nq,nv=e.m.nv,nu=e.m.nu);(out/'scene.xml').write_text(e.xml);m=e.panel_metrics();flip_last=trace[-1]['metrics'].get('flip') if trace else None;flip_pass=bool(a.flip and finished and error is None and flip and flip.state=='done' and pre_flip['frame']['frame_geometry_ok'] and pre_flip['xy_error_m']<.006 and flip_last['panel_up_dot']<-.98 and max(flip_last['position_errors_m'])<.005 and max(flip_last['angle_errors_rad'])<np.deg2rad(3) and e.nails is not None and e.nails.metrics()['intact']==8 and flip_last['minimum_height_above_fixture_m']>.02);save(out/'result.json',{'finished':finished,'success':flip_pass if a.flip else bool(finished and error is None and len(events)==8 and m['xy_error_m']<.006 and abs(m['height_error_m'])<.003 and m['tilt_deg']<1 and m['frame']['frame_geometry_ok']),'error':error,'metrics':m,'events':events,'simulated_s':e.d.time,'wall_s':time.monotonic()-start,'continuous':not bool(a.resume_diagnostic),'diagnostic_checkpoint':a.resume_diagnostic,'camera_control_after_checkpoint':a.camera_after_resume,'actual_astra':False,'privileged_diagnostic':a.oracle,'flip_profile':a.flip_profile,'shoulder_half_span_m':a.shoulder_half_span,'fixture_access_slots':a.flip and a.flip_profile=='clearance-stack','dt':a.dt,'north_lift_height_override_m':a.north_lift_height,'physical_validation':False,'pre_flip_assembly':pre_flip,'flip_requested':a.flip,'flip_finished':bool(flip and flip.state=='done'),'flip_last_metrics':trace[-1]['metrics'].get('flip') if trace else None,'fastener_physics':None if e.nails is None else e.nails.metrics()});print('saved',out,flush=True)
if __name__=='__main__':main()
