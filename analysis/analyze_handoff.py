import argparse,gzip,json
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(); parser.add_argument("root"); ROOT=Path(parser.parse_args().root)
out={}
for name in ['delayed_frame_before','delayed_frame_after']:
 folder=ROOT/name
 result=json.loads((folder/'evaluator/result.json').read_text())
 with gzip.open(folder/'evaluator/trace.jsonl.gz','rt') as f: rows=[json.loads(x) for x in f]
 times=np.array([x['time'] for x in rows]); details=[]
 for entry in result['history']:
  if entry['command']['skill']!='position_member':continue
  i=entry['command']['index'];t=entry['completed_sim_s'];a=next(j for j,row in enumerate(rows) if row['time']>=t-1e-8 and row['stage']=='done')
  release=next(x for x in result['history'] if x['command']['skill']=='release_member' and x['command']['index']==i)
  b=int(np.argmin(abs(times-release['issued_sim_s'])))
  pos=lambda row:np.array(row['metrics']['frame']['wood_evaluator_only'][i]['center'])
  details.append({'member':i,'command_jump_mm':float(np.linalg.norm(np.array(rows[a]['command'][0])-np.array(rows[a-1]['command'][0]),axis=1).max()*1000),'wait_seconds':float(times[b]-times[a]),'object_displacement_wait_mm':float(np.linalg.norm(pos(rows[b])-pos(rows[a]))*1000),'placement_xy_at_done_mm':rows[a]['metrics']['frame']['position_error_m'][i]*1000,'placement_xy_before_release_mm':rows[b]['metrics']['frame']['position_error_m'][i]*1000})
 out[name]={'handoffs':details,'final_frame_geometry_ok':rows[-1]['metrics']['frame']['frame_geometry_ok'],'final_position_errors_mm':(np.array(rows[-1]['metrics']['frame']['position_error_m'])*1000).tolist(),'final_yaw_errors_deg':rows[-1]['metrics']['frame']['yaw_error_deg'],'peak_penetration_mm':result['peak_penetration_m']*1000,'stop':'intentional stop after four members; no full-task success claim'}
(ROOT/'handoff_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
