"""Frozen-budget zero-initialized residual experiment; preserves the v1 pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np

from .rail_rl import OUT as OLD, RailEnv

OUT = OLD.parent / 'rl_zero_init'
STEPS = 262144
SCALE = .1
CHECKPOINTS = (131072, 262144)
VALIDATION = [('id', s) for s in range(30000, 30020)]
TEST = [('id', s) for s in range(40000, 40020)] + [('ood', s) for s in range(50000, 50020)]
INFO_KEYS = ('success', 'lift_success', 'max_hold_seconds', 'max_lowest_point_m', 'max_penetration_m')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2))


def params():
    return json.loads((OLD / 'base_params.json').read_text())


class BoundedResidual(gym.ActionWrapper):
    """Map policy actions to corrections of at most 10% of each motor range."""
    def action(self, action):
        return SCALE * np.clip(np.asarray(action), -1., 1.)


def factory(seed):
    def make():
        from stable_baselines3.common.monitor import Monitor
        e = BoundedResidual(RailEnv('residual', params()))
        e.reset(seed=seed)
        return Monitor(e, info_keywords=INFO_KEYS)
    return make


def zero_output(model):
    import torch
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    # The exploration schedule is explicit, not an additional learned parameter.
    model.policy.log_std.requires_grad_(False)


def exploration_sigma(steps):
    # Effective action-space noise starts at .005 and reaches .03, capped at .1.
    return .05 + .25 * min(1., steps / 65536)


def new_model(vec, seed):
    from stable_baselines3 import PPO
    model = PPO('MlpPolicy', vec, learning_rate=1e-4, n_steps=256,
                batch_size=256, n_epochs=5, gamma=.995, gae_lambda=.95,
                ent_coef=0., seed=seed, device='cpu', verbose=0,
                policy_kwargs={'net_arch':[128,128], 'log_std_init':float(np.log(.05))})
    zero_output(model)
    return model


def train(seed):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    torch.set_num_threads(1)
    folder = OUT / f'seed{seed}'
    folder.mkdir()
    vec = VecNormalize(SubprocVecEnv([factory(seed*100+i) for i in range(4)],
                                     start_method='spawn'), norm_obs=True, norm_reward=False)
    model = new_model(vec, seed)
    obs = vec.reset()
    initial_action = model.predict(obs, deterministic=True)[0]
    assert np.array_equal(initial_action, np.zeros_like(initial_action))
    dump(folder / 'initialization.json', {'maximum_absolute_deterministic_action':float(abs(initial_action).max()),
         'output_weight_max':float(model.policy.action_net.weight.abs().max()),
         'output_bias_max':float(model.policy.action_net.bias.abs().max()),
         'raw_noise_std':exploration_sigma(0), 'effective_noise_std':SCALE*exploration_sigma(0)})
    model.save(folder / 'initial'); vec.save(folder / 'initial_normalization.pkl')
    start = time.monotonic()

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__(); self.rows=[]

        def _on_rollout_start(self):
            with torch.no_grad():
                self.model.policy.log_std.fill_(np.log(exploration_sigma(self.num_timesteps)))

        def _on_step(self):
            for info in self.locals.get('infos', []):
                if 'episode' in info:
                    row={'step':self.num_timesteps, **info['episode']}
                    self.rows.append(row)
                    with (folder / 'episodes.jsonl').open('a') as f:
                        f.write(json.dumps(row)+'\n')
            if self.num_timesteps % 8192 == 0:
                recent=self.rows[-30:]
                row={'steps':self.num_timesteps, 'wall_seconds':time.monotonic()-start,
                     'recent_lift':float(np.mean([r['lift_success'] for r in recent])) if recent else 0,
                     'recent_full_success':float(np.mean([r['success'] for r in recent])) if recent else 0,
                     'effective_noise_std':SCALE*exploration_sigma(self.num_timesteps)}
                dump(folder/'progress.json', row); print(json.dumps(row), flush=True)
            return True

    callback=Progress()
    # Save after the optimizer has consumed each segment's final rollout.
    for checkpoint in CHECKPOINTS:
        model.learn(total_timesteps=checkpoint-model.num_timesteps, callback=callback,
                    reset_num_timesteps=False)
        model.save(folder / f'step{checkpoint}')
        vec.save(folder / f'step{checkpoint}_normalization.pkl')
    dump(folder / 'training_complete.json', {'steps':model.num_timesteps,
         'wall_seconds':time.monotonic()-start, 'episodes':len(callback.rows),
         'training_lift_successes':sum(bool(r['lift_success']) for r in callback.rows),
         'training_full_successes':sum(bool(r['success']) for r in callback.rows)})
    vec.close()


def case(job):
    """Baseline, zero-policy equivalence, and paired exploration-only probes."""
    condition, split, seed = job
    e=RailEnv('residual', params(), ood=split=='ood')
    obs,_=e.reset(seed=seed)
    rng=np.random.default_rng(seed+700000)
    baseline=None; model=None; total=0.; max_difference=0.
    if condition == 'zero_policy':
        import torch
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        torch.set_num_threads(1)
        baseline=RailEnv('base',params(),ood=split=='ood'); baseline.reset(seed=seed)
        vec=VecNormalize(DummyVecEnv([factory(0)]),norm_obs=True,norm_reward=False)
        vec.training=False; model=new_model(vec,0)
    for _ in range(e.max_steps):
        if condition == 'legacy_noise':
            action=np.clip(rng.normal(0, np.exp(-1.5),26),-1,1)
        elif condition == 'small_noise_start':
            action=SCALE*np.clip(rng.normal(0,exploration_sigma(0),26),-1,1)
        elif condition == 'small_noise_final':
            action=SCALE*np.clip(rng.normal(0,exploration_sigma(STEPS),26),-1,1)
        elif model is not None:
            action=SCALE*model.predict(vec.normalize_obs(obs[None]),deterministic=True)[0][0]
            assert np.array_equal(action,np.zeros_like(action))
        else:
            action=np.zeros(26)
        obs,r,done,truncated,info=e.step(action); total+=r
        if baseline is not None:
            baseline.step(np.zeros(26))
            max_difference=max(max_difference,float(abs(e.sim.data.qpos-baseline.sim.data.qpos).max()),
                               float(abs(e.sim.data.qvel-baseline.sim.data.qvel).max()))
            assert max_difference == 0.
        if done or truncated: break
    if model is not None:vec.close()
    return {'condition':condition, 'split':split, 'eval_seed':seed, 'return':total,
            'max_baseline_state_difference':max_difference if baseline is not None else None, **info}


def evaluate(job):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    torch.set_num_threads(1)
    label, cases = job
    seed_text, step_text=label.split('/')
    folder=OUT/seed_text
    rows=[]
    for split in sorted(set(s for s,_ in cases)):
        env=BoundedResidual(RailEnv('residual',params(),ood=split=='ood'))
        vec=VecNormalize.load(folder/f'{step_text}_normalization.pkl',DummyVecEnv([lambda:env]))
        vec.training=False;vec.norm_reward=False
        model=PPO.load(folder/step_text,device='cpu')
        for _,seed in [(s,n) for s,n in cases if s==split]:
            obs,_=env.reset(seed=seed);total=0
            for _ in range(env.unwrapped.max_steps):
                a=model.predict(vec.normalize_obs(obs[None]),deterministic=True)[0][0]
                obs,r,done,truncated,info=env.step(a);total+=r
                if done or truncated:break
            rows.append({'condition':label,'split':split,'eval_seed':seed,'return':total,**info})
        vec.close()
    return rows


def counts(rows):
    return {'episodes':len(rows),'lift':sum(bool(r['lift_success']) for r in rows),
            'full_success':sum(bool(r['success']) for r in rows),
            'strict_episode_lift':sum(r['lift_success'] and r['max_penetration_m']<.001 for r in rows),
            'mean_peak_penetration_mm':float(np.mean([r['max_penetration_m']*1000 for r in rows])),
            'max_penetration_mm':max(r['max_penetration_m']*1000 for r in rows)}


def quality(rows):
    c=counts(rows)
    return (c['full_success'], c['lift'])


def select(baseline, candidates):
    """Retain zero correction on a tie; selection sees validation cases only."""
    chosen='baseline';best=quality(baseline)
    for label,rows in candidates.items():
        score=quality(rows)
        if score>best:
            chosen,best=label,score
    return chosen


def run():
    if OUT.exists():raise RuntimeError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    source=Path(__file__).resolve().parent
    manifest={'experiment':'zero-initialized bounded residual PPO pilot',
        'created_unix':time.time(),'train_seeds':[0,1],'steps_per_seed':STEPS,
        'residual_scale':SCALE,'initialization':'exactly zero output weights and bias',
        'noise':'frozen policy std ramps .05 -> .30 over first 65536 steps; effective std .005 -> .030 after residual scaling',
        'learning_rate':1e-4,'entropy_coefficient':0,'checkpoints':list(CHECKPOINTS),
        'validation_cases':VALIDATION,'fresh_test_cases':TEST,
        'selection':'per training seed, choose baseline/mid/final by validation (full success count, lift count), lexicographic; strict improvement required; ties retain earlier candidate with baseline first',
        'exploration_cases':'20 validation cases with paired Gaussian draws; no policy updates',
        'physics':'unchanged from v1, including unresolved penetration; hold criterion is not episode-wide validity',
        'reward':'unchanged; residual magnitude penalties apply to physical scaled correction',
        'historical_test':'v1 test cases not reused in this evaluation',
        'source_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['environment.py','rail_rl.py','rail_residual_v2.py']},
        'base_params_sha256':hashlib.sha256((OLD/'base_params.json').read_bytes()).hexdigest()}
    dump(OUT/'manifest.json',manifest)
    start=time.monotonic()
    with ProcessPoolExecutor(4,mp_context=mp.get_context('spawn')) as pool:
        # Full-episode exact equality checks use both standard and harder scenes.
        checks=list(pool.map(case,[('zero_policy','id',30000),('zero_policy','id',30001),('zero_policy','ood',60000)]))
        dump(OUT/'zero_initialization_checks.json',checks)
        print('Exact-zero policy: three full episodes match baseline state for state.',flush=True)
        probes=[]
        for condition in ('baseline','legacy_noise','small_noise_start','small_noise_final'):
            rows=list(pool.map(case,[(condition,s,n) for s,n in VALIDATION]))
            probes+=rows;dump(OUT/'exploration_only.json',probes)
            print(json.dumps({'exploration_condition':condition,**counts(rows)}),flush=True)
    processes=[]
    for seed in (0,1):
        f=(OUT/f'train_seed{seed}.log').open('w')
        p=subprocess.Popen([sys.executable,'-u','-m','experiments.frame_assembly_v2.rail_residual_v2','train','--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT)
        processes.append((seed,p,f))
    failures=[]
    for seed,p,f in processes:
        code=p.wait();f.close()
        print(json.dumps({'trained_seed':seed,'exit_code':code}),flush=True)
        if code:failures.append(seed)
    if failures:raise RuntimeError(f'Training failed: {failures}')
    baseline=[r for r in probes if r['condition']=='baseline']
    labels=[f'seed{seed}/step{step}' for seed in (0,1) for step in CHECKPOINTS]
    with ProcessPoolExecutor(4,mp_context=mp.get_context('spawn')) as pool:
        validation=dict(zip(labels,pool.map(evaluate,[(label,VALIDATION) for label in labels])))
        dump(OUT/'validation.json',{'baseline':baseline,**validation})
        chosen={f'seed{seed}':select(baseline,{k:v for k,v in validation.items() if k.startswith(f'seed{seed}/')}) for seed in (0,1)}
        dump(OUT/'selection.json',chosen)
        print('Validation selection '+json.dumps(chosen),flush=True)
        # Choices are saved before any fresh test outcomes are calculated.
        test_base=list(pool.map(case,[('baseline',s,n) for s,n in TEST]))
        test_labels=sorted(set([f'seed{s}/step{STEPS}' for s in (0,1)]+[v for v in chosen.values() if v!='baseline']))
        test=dict(zip(test_labels,pool.map(evaluate,[(label,TEST) for label in test_labels])))
        dump(OUT/'test.json',{'baseline':test_base,**test})
    report={'wall_seconds':time.monotonic()-start,'training':[json.loads((OUT/f'seed{s}/training_complete.json').read_text()) for s in (0,1)],
            'exploration':{c:counts([r for r in probes if r['condition']==c]) for c in ('baseline','legacy_noise','small_noise_start','small_noise_final')},
            'validation':{k:counts(v) for k,v in {'baseline':baseline,**validation}.items()},
            'selected':chosen,'test':{k:{s:counts([r for r in rows if r['split']==s]) for s in ('id','ood')} for k,rows in {'baseline':test_base,**test}.items()}}
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest()==h for name,h in manifest['source_sha256'].items()}
    hashes['base_params']=hashlib.sha256((OLD/'base_params.json').read_bytes()).hexdigest()==manifest['base_params_sha256']
    assert all(hashes.values()),hashes
    report['frozen_hashes_verified']=hashes
    dump(OUT/'results.json',report)
    print('COMPLETE '+json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run','train'])
    parser.add_argument('--seed',type=int,default=0);args=parser.parse_args()
    if args.operation=='train':train(args.seed)
    else:run()
