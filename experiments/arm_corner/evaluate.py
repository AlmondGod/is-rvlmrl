"""Frozen small reserved-seed evaluation, separate from actual Astra decisions."""
import hashlib,json,shutil,sys
from pathlib import Path
import mujoco,numpy as np
from .run import run

def hashes():
 paths=list(Path('experiments/arm_corner').glob('*.py'))+list(Path('experiments/arm_corner').glob('*.json'))+[Path('third_party/3-PoinTr/simulation_assets/ufactory_xarm7/xarm7.xml')]
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def main():
 root=Path('outputs/arm_corner/frozen_eval_v1');root.mkdir(exist_ok=False);h=hashes();(root/'freeze.json').write_text(json.dumps({'hashes':h,'mujoco':mujoco.__version__,'python':sys.version},indent=2))
 for p in h:
  dest=root/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 cases=[('dev_final',dict(seed=0,condition='camera')),('dev_half',dict(seed=0,condition='camera',dt=.0005)),('oracle_final',dict(seed=0,condition='oracle'))]+[(f'reserved_{s}',dict(seed=s,condition='camera')) for s in range(7100,7105)]
 rows=[]
 for name,args in cases:
  assert hashes()==h,'Sources changed after freeze'
  r=run(root/name,**args);rows.append({'name':name,**{k:r[k] for k in ('seed','dt','strict_success','failure','pickup','released_stable_s','metrics','simulated_s')}});(root/'summary.json').write_text(json.dumps(rows,indent=2))
 a,b=[json.loads((root/n/'result.json').read_text()) for n in ('dev_final','dev_half')];delta=np.linalg.norm(np.array(a['metrics']['wood_evaluator_only'][0]['center'])-np.array(b['metrics']['wood_evaluator_only'][0]['center']));(root/'numerical.json').write_text(json.dumps({'same_success_classification':a['strict_success']==b['strict_success'],'final_member_position_difference_m':float(delta),'source_unchanged':hashes()==h},indent=2))
if __name__=='__main__':main()
