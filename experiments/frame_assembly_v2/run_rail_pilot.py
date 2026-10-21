"""Bounded, restartable local experiment runner; creates no cloud resources."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .rail_rl import OUT


def pair(commands):
    processes=[]
    for label,args in commands:
        log=(OUT/f'{label}.log').open('w')
        p=subprocess.Popen([sys.executable,'-u','-m','experiments.frame_assembly_v2.rail_experiment',*args],stdout=log,stderr=subprocess.STDOUT)
        processes.append((label,p,log))
    failures=[]
    for label,p,log in processes:
        code=p.wait();log.close()
        print(json.dumps({'job':label,'exit_code':code}),flush=True)
        if code:failures.append(label)
    if failures:raise RuntimeError(f'Jobs failed: {failures}; inspect their logs')


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    if not (OUT/'base_seed0/evaluation.json').exists():
        raise RuntimeError('Finish the corrected non-learning baseline evaluation first')
    directory=Path(__file__).resolve().parent
    manifest={'created_unix':time.time(),'experiment':'one-rail lift >=5cm, stable hold >=2s, translate12cm, place and release >=1s','run_kind':'bounded exploratory pilot','train_seeds':[0,1],'steps_per_seed_per_method':262144,'methods':['optimized feedback baseline','residual PPO','direct PPO'],'historical_astra':'Three prior trials are qualitative context, not a matched full-task test','base_search_rollouts':100,'base_search_control_steps':55000,'base_search_conditions':'single nominal case; no held-out evaluation cases used','shared_prior_information':'Both learned variants observe the nominal command; only residual PPO adds it to its action','same_motor_rate_limits':True,'same_reward_observations_and_training_distribution':True,'base_frozen':True,'evaluation':'20 paired ID seeds and20 paired OOD seeds, fixed final-budget checkpoint','hand_friction':'nominal hand friction multiplied by wood sliding coefficient /0.55; preserves original nominal physics','source_sha256':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ['environment.py','rail_rl.py','rail_experiment.py']},'base_params_sha256':hashlib.sha256((OUT/'base_params.json').read_bytes()).hexdigest()}
    (OUT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    start=time.monotonic()
    for seed in (0,1):
        pair([(f'train_{mode}_{seed}',['train','--mode',mode,'--seed',str(seed),'--steps','262144','--workers','4']) for mode in ('residual','direct') if not (OUT/f'{mode}_seed{seed}/training_complete.json').exists()])
        pair([(f'evaluate_{mode}_{seed}',['evaluate','--mode',mode,'--seed',str(seed),'--episodes','20',*(['--video'] if seed==0 else [])]) for mode in ('residual','direct')])
    reports=[]
    for name in ('base_seed0','residual_seed0','direct_seed0','residual_seed1','direct_seed1'):
        reports.append({'run':name,**json.loads((OUT/name/'evaluation.json').read_text())})
    (OUT/'pilot_results.json').write_text(json.dumps({'wall_seconds':time.monotonic()-start,'manifest':manifest,'runs':reports},indent=2))
    print('PILOT COMPLETE',flush=True)


if __name__=='__main__':main()
