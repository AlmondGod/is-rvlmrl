"""Independent evaluator audit of a recorded continuous run (never a policy input)."""
import json,sys,importlib.util
from pathlib import Path
import numpy as np
from .environment import PANEL_SIZE,PANEL_START,SIZES,CORNER_PAIRS
spec=importlib.util.spec_from_file_location('ray_helpers','outputs/frame_assembly/audit_v2.py');helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
def audit(folder):
 folder=Path(folder);r=json.loads((folder/'result.json').read_text());tr=json.loads((folder/'trace.json').read_text());ts=np.array([t['time'] for t in tr]);dt=np.r_[.02,np.diff(ts)];metrics=[t['metrics'] for t in tr];last=r['metrics'];grip=np.array([m['forces']['grip'] for m in metrics]);center=np.array([m['center_evaluator_only'] for m in metrics]);rots=np.array([m['rotation_evaluator_only'] for m in metrics]);rack=np.array([m['forces']['rack'] for m in metrics]);support=np.array([m['forces']['frame'] for m in metrics]);other=np.array([m['forces']['other'] for m in metrics]);speed=np.r_[0,np.linalg.norm(np.diff(center,axis=0),axis=1)/dt[1:]]
 bottom=center[:,2]-np.sum(abs(rots[:,2,:])*PANEL_SIZE/2,axis=1)
 lifted=helpers.streak((bottom>PANEL_START[2]-PANEL_SIZE[2]/2+.02)&(np.min(grip,axis=1)>10)&(rack<.01),dt)
 good=np.array([m['frame']['frame_geometry_ok'] and m['xy_error_m']<.006 and abs(m['height_error_m'])<.003 and m['tilt_deg']<1 for m in metrics])&(abs(np.arctan2(rots[:,1,0],rots[:,0,0]))<np.deg2rad(1))&(np.max(grip,axis=1)<.1)&(rack<.01)&(other<.1)&(support>50)&(speed<.005)
 stable=0.
 for ok,duration in zip(good[::-1],dt[::-1]):
  if not ok:break
  stable+=duration
 rays=[]
 for ev in r['events']:
  k=int(np.argmin(abs(ts-ev['time'])));rob=tr[k]['robot'];origin=np.array(rob['tcp'][2]);direction=np.array(rob['rotation'][2])[:,2];members=CORNER_PAIRS[ev['index']] if ev['kind']=='frame' else [ev['index']];crossings=[]
  if ev['kind']=='panel':crossings.append(helpers.ray_box(origin,direction,center[k],rots[k],PANEL_SIZE))
  for i in members:
   p=metrics[k]['frame']['wood_evaluator_only'][i];crossings.append(helpers.ray_box(origin,direction,np.array(p['center']),np.array(p['rotation']),SIZES[i]))
  rays.append({'kind':ev['kind'],'index':ev['index'],'intersections_m':crossings,'crosses_both':all(x is not None for x in crossings)})
 weight=last['panel_mass_kg']*9.81;balance=abs(last['forces']['frame']-weight)/weight;peak=last['frame']['peak_penetration_m'];finite=last['frame']['finite'] and not any(last['frame']['warnings']);unique={(e['kind'],e['index']) for e in r['events']};continuous=r['continuous'] and not r['diagnostic_checkpoint'] and ts[0]<.021 and np.max(abs(np.diff(ts)-.02))<1e-7
 passed=bool(r['success'] and continuous and len(unique)==len(r['events'])==8 and stable>=2.9 and lifted>=.5 and balance<.05 and peak<.0005 and finite and all(x['crosses_both'] for x in rays))
 return dict(passes=passed,continuous_from_loose_parts=bool(continuous),privileged_diagnostic=r['privileged_diagnostic'],actual_astra=r['actual_astra'],runner_success=r['success'],error=r['error'],frame_member_xy_errors_mm=(np.array(last['frame']['position_error_m'])*1000).tolist(),panel_xy_error_mm=last['xy_error_m']*1000,panel_height_error_mm=last['height_error_m']*1000,panel_tilt_deg=last['tilt_deg'],peak_penetration_mm=peak*1000,independent_sheet_lift_s=lifted,released_stable_s=stable,panel_weight_N=weight,panel_support_N=last['forces']['frame'],relative_force_balance_error=balance,finite_no_solver_warnings=finite,virtual_nail_rays=rays,simulated_s=r['simulated_s'],wall_s=r['wall_s'],physical_validation=False)
if __name__=='__main__':
 for folder in sys.argv[1:]:
  result=audit(folder);Path(folder,'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
