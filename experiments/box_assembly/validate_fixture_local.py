"""Local numerical regression of full free-body box fixture, not policy success."""
from dataclasses import replace
import argparse,json
from pathlib import Path
import numpy as np
from .design import Config
from .fastening_environment import FasteningBoxAssembly


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 rows=[]
 for dt in (.001,.0005,.00025):
  e=FasteningBoxAssembly(replace(Config(),timestep=dt),allow_local=True)
  assert e.model.neq==0
  for assembled in (False,True):
   e.reset(seed=1729,diagnostic_assembled=assembled)
   for _ in range(200):e.step(e.wrist_target.copy(),e.wrist_quaternion.copy(),e.finger_target.copy())
   settled=e.evaluator_metrics()
   e.fixture_release(True)
   for _ in range(180):e.step(e.wrist_target.copy(),e.wrist_quaternion.copy(),e.finger_target.copy())
   released=e.evaluator_metrics()
   row={'dt':dt,'diagnostic_assembled':assembled,'settled':settled,'after_fixture_retraction':released}
   rows.append(row);(a.out/'report.json').write_text(json.dumps({'cases':rows,'complete':False},indent=2)+'\n')
   assert settled['finite_state'] and released['finite_state'] and not any(settled['warnings']+released['warnings'])
   assert released['peak_episode_penetration_m']<e.config.success_penetration_tolerance,row
   if assembled:assert max(x['position_m'] for x in settled['pose_errors'].values())<.0015,row
   print(json.dumps({'dt':dt,'assembled':assembled,'peak_penetration_m':released['peak_episode_penetration_m'],'settled_max_position_error_m':max(x['position_m'] for x in settled['pose_errors'].values()),'retracted':released['fixture_retracted']}),flush=True)
 (a.out/'report.json').write_text(json.dumps({'cases':rows,'complete':True,'conclusion':'fixture numerical regression only; diagnostic resets are not assembly success'},indent=2)+'\n')

if __name__=='__main__':main()
