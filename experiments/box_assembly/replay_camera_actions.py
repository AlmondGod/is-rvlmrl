"""Numerical diagnostic: replay recorded motor actions, NOT new model decisions."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from .design import Config
from .fastening_environment import FasteningBoxAssembly
from .astra_camera_bridge import xml


def main():
 p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False)
 requests=sorted((a.recording/'evaluator').glob('request_*.json'),key=lambda p:p.stat().st_mtime_ns)
 actions=[json.loads(p.read_text())['action'] for p in requests]
 seed=json.loads((a.recording/'run_manifest.json').read_text())['seed']
 results=[]
 for dt in (.0005,.00025):
  e=FasteningBoxAssembly(replace(Config(),timestep=dt),allow_local=True,xml_builder=xml);e.reset(seed=seed)
  for _ in range(50):e.step(e.wrist_target.copy(),e.wrist_quaternion.copy(),e.finger_target.copy())
  pose={'position':e.wrist_target.copy(),'quaternion':e.wrist_quaternion.copy(),'fingers':e.finger_target.copy()};rows=[]
  for index,action in enumerate(actions,1):
   if action.get('stop'):break
   pose.update({k:action[k] for k in pose if k in action})
   if 'fixture_release' in action:e.fixture_release(bool(action['fixture_release']))
   for _ in range(max(1,round(float(action.get('duration_s',.5))/e.config.control_period))):
    metrics=e.step(pose['position'],pose['quaternion'],pose['fingers'],tools=action.get('tools',{}))
    if not metrics['finite_state'] or any(metrics['warnings']):raise RuntimeError('Invalid numerical replay')
   rows.append({'action':index,'metrics':metrics,'wrist_position':e.data.xpos[e.wid].tolist(),'wrist_quaternion':e.data.xquat[e.wid].tolist()})
  results.append({'timestep_s':dt,'actions':rows})
  (a.out/'report.json').write_text(json.dumps({'interpretation':'Recorded open-loop motor replay only; not a camera-controller reevaluation or physical validation','results':results},indent=2)+'\n')
  print(json.dumps({'timestep_s':dt,'actions':len(rows),'penetration_m':metrics['peak_episode_penetration_m'],'dry_fit':metrics['dry_fit_success']}),flush=True)

if __name__=='__main__':main()
