"""Independent recorded-state audit of continuous assembly + retention test."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,mujoco
from .flip_evaluator import reference,measure,passes
from .trace_io import iter_trace

def main():
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--checkpoint',action='store_true');a=ap.parse_args();p=Path(a.run);result=json.loads((p/'result.json').read_text());trace=[{'stage':x.get('stage','flip: '+x.get('phase',''))} for x in iter_trace(p/'trace.json')];rows=np.load(p/'raw.npz')['rows'];m=mujoco.MjModel.from_xml_path(str(p/'scene.xml'));d=mujoco.MjData(m);e=SimpleNamespace(m=m,d=d,woods=[m.body(f'wood{i}').id for i in range(4)],panel=m.body('panel').id)
 fasteners=result['last']['fasteners'] if a.checkpoint else result['fastener_physics'];start=next((i for i,x in enumerate(trace) if x['stage'].startswith('flip:')),None);last=None;hold=0.;other_contact=False;robot_contacts={};airborne_position=0.;airborne_angle=0.
 def load(i):
  d.qpos[:]=rows[i,1:1+m.nq];d.qvel[:]=rows[i,1+m.nq:1+m.nq+m.nv];mujoco.mj_forward(m,d)
 if start is not None:
  load(start);ref=reference(e)
  for i in range(start,len(rows)):
   load(i);flight=measure(e,ref)
   if flight['minimum_height_above_fixture_m']>.02:
    airborne_position=max(airborne_position,max(flight['position_errors_m']));airborne_angle=max(airborne_angle,max(flight['angle_errors_rad']))
   for c in d.contact:
    ns=[m.geom(g).name or '' for g in (c.geom1,c.geom2)]
    if c.dist<-.000005 and any(n.startswith('wood') or n=='panel_geom' for n in ns) and any(n.startswith(('a0_','a1_','a2_')) and 'finger_pad' not in n for n in ns):
     key=' / '.join(ns);record=robot_contacts.setdefault(key,dict(first_time_s=float(rows[i,0]),samples=0,maximum_depth_m=0.));record['samples']+=1;record['maximum_depth_m']=max(record['maximum_depth_m'],float(-c.dist))
  for i in range(max(start,len(rows)-300),len(rows)):
   load(i);last=measure(e,ref);contacts=[]
   for c in d.contact:
    names=[m.geom(g).name or '' for g in (c.geom1,c.geom2)]
    work=lambda n:n.startswith('wood') or n=='panel_geom'
    if c.dist<=0 and any(work(n) for n in names) and not all(work(n) for n in names) and not any('finger_pad' in n for n in names):contacts.append(names)
   other_contact=bool(contacts);good=passes(last,fasteners['intact'],True) and not other_contact
   hold=hold+.02 if good else 0.
 pre=result.get('pre_flip_assembly');pre_ok=bool(pre and pre['frame']['frame_geometry_ok'] and pre['xy_error_m']<.006 and abs(pre['height_error_m'])<.003 and pre['tilt_deg']<1)
 peak=result['last']['peak_penetration_m'] if a.checkpoint else result['metrics']['frame']['peak_penetration_m'];continuous=bool(result.get('continuous',False) and not result.get('diagnostic_checkpoint') and rows[0,0]<.021 and np.max(abs(np.diff(rows[:,0])-.02))<1e-7)
 flip_finished=result['last']['phase']=='done' if a.checkpoint else result.get('flip_finished',False)
 passed=bool((a.checkpoint or (continuous and pre_ok)) and flip_finished and result['error'] is None and hold>=4.9 and peak<=.0005 and not robot_contacts and airborne_position<.005 and airborne_angle<np.deg2rad(3))
 out=dict(passes=passed,maximum_airborne_relative_position_error_mm=airborne_position*1000,maximum_airborne_relative_angle_error_deg=float(np.degrees(airborne_angle)),unexpected_robot_contacts_20ms_sampling=robot_contacts,checkpoint_diagnostic_only=a.checkpoint,continuous_from_loose_parts=continuous,pre_flip_assembly_passes=pre_ok,flip_started=start is not None,flip_finished=flip_finished,inverted_retained_unsupported_seconds=hold,last_flip_metrics=last,other_support_contact=other_contact,peak_penetration_mm=peak*1000,fasteners=fasteners,error=result['error'],actual_astra=result['actual_astra'],physical_validation=False)
 (p/'flip_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
