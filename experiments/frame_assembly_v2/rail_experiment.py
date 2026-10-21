"""Optimize a nominal grasp, train matched PPO pilots, and evaluate held-out seeds."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing as mp
import time

import numpy as np

from .rail_rl import DEFAULT, OUT, RailEnv, rollout


def decode(x):
    p=deepcopy(DEFAULT)
    p['offset']=x[:3].tolist()
    q=np.array(p['close']);q[:4]=x[3:7]
    for i in (4,8,12,16):q[i]=x[7];q[i+2]=x[8];q[i+3]=x[9]
    p['close']=q.tolist()
    return p


def candidate(x):
    result=rollout(decode(np.array(x)),randomize=False,max_steps=550)
    score=10*min(result['max_lowest_point_m'],.1)+3*result['max_hold_seconds']+20*result['lift_success']-.1*max(0,result['max_penetration_m']*1000-1)+.02*result['return']
    return dict(score=score,params=decode(np.array(x)),metrics=result)


def optimize():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    mean=np.r_[DEFAULT['offset'],DEFAULT['close'][:4],[.7,1.3,.7]]
    std=np.array([.018,.035,.014,.4,.3,.4,.4,.35,.4,.35])
    low=np.array([.085,-.055,.035,-.4,-1.2,-.8,-.8,-.2,.0,-.2])
    high=np.array([.155,.10,.11,1.28,.5,1.5,1.5,1.5,1.9,1.4])
    all_results=[];best=None;start=time.monotonic()
    with ProcessPoolExecutor(4,mp_context=mp.get_context('spawn')) as pool:
        for generation in range(5):
            xs=np.clip(rng.normal(mean,std,(20,len(mean))),low,high)
            xs[0]=mean
            results=list(pool.map(candidate,xs.tolist()))
            ranked=np.argsort([r['score'] for r in results])[::-1]
            if best is None or results[ranked[0]]['score']>best['score']:best=results[ranked[0]]
            elite=xs[ranked[:5]];mean=.7*elite.mean(0)+.3*mean;std=np.maximum(elite.std(0),std*.4)
            all_results+=results
            (OUT/'base_search.json').write_text(json.dumps({'generations_completed':generation+1,'rollouts':all_results,'best':best,'wall_seconds':time.monotonic()-start},indent=2))
            print(json.dumps({'generation':generation+1,'best_score':best['score'],'metrics':best['metrics'],'wall_seconds':time.monotonic()-start}),flush=True)
    (OUT/'base_params.json').write_text(json.dumps(best['params'],indent=2))


def make_env(mode,params,seed):
    def create():
        from stable_baselines3.common.monitor import Monitor
        e=RailEnv(mode,params)
        e.reset(seed=seed)
        return Monitor(e,info_keywords=('success','lift_success','max_hold_seconds','max_lowest_point_m','max_penetration_m'))
    return create


def train(mode,seed,steps,workers):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
    torch.set_num_threads(1)
    params=json.loads((OUT/'base_params.json').read_text())
    folder=OUT/f'{mode}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    config={'mode':mode,'seed':seed,'requested_steps':steps,'workers':workers,'base_params':params,'algorithm':'PPO','network':[128,128],'observations':'identical 164-dimensional privileged state; includes nominal action for BOTH modes','control_hz':50,'physics_hz':1000,'initial_state':'full approach, rails resting on bench; no pregrasp resets','eval_seeds':'10000-10019 ID;20000-20019 OOD','selection':'final fixed-budget checkpoint, no test-set tuning','training_randomization':{'position_m':.015,'yaw_rad':.1,'mass_factor':[.8,1.2],'friction':[.4,.8]},'ood_randomization':{'position_m':.03,'yaw_rad':.2,'mass_factor':[.6,1.4],'friction':[.3,.9],'geometry_scale':[.9,1.1]}}
    (folder/'config.json').write_text(json.dumps(config,indent=2))
    v=SubprocVecEnv([make_env(mode,params,seed*100+i) for i in range(workers)],start_method='spawn')
    v=VecNormalize(v,norm_obs=True,norm_reward=False,clip_obs=10.)
    class Progress(BaseCallback):
        def __init__(self):super().__init__();self.start=time.monotonic();self.episodes=[];self.last=0
        def _on_step(self):
            for info in self.locals.get('infos',[]):
                if 'episode' in info:
                    row={'step':self.num_timesteps,**info['episode']}
                    self.episodes.append(row)
                    with (folder/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if self.num_timesteps-self.last>=8192:
                self.last=self.num_timesteps
                recent=self.episodes[-30:]
                row={'steps':self.num_timesteps,'wall_seconds':time.monotonic()-self.start,'episodes':len(self.episodes),'recent_success':float(np.mean([r['success'] for r in recent])) if recent else 0,'recent_lift':float(np.mean([r['lift_success'] for r in recent])) if recent else 0,'recent_max_height':float(np.mean([r['max_lowest_point_m'] for r in recent])) if recent else 0}
                (folder/'progress.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
                if self.num_timesteps%32768==0:self.model.save(folder/'checkpoint');v.save(folder/'normalization.pkl')
            return True
    model=PPO('MlpPolicy',v,learning_rate=3e-4,n_steps=256,batch_size=256,n_epochs=5,gamma=.995,gae_lambda=.95,ent_coef=.002,policy_kwargs={'net_arch':[128,128],'log_std_init':-1.5},seed=seed,device='cpu',verbose=0)
    start=time.monotonic();model.learn(total_timesteps=steps,callback=Progress());model.save(folder/'final');v.save(folder/'normalization.pkl')
    (folder/'training_complete.json').write_text(json.dumps({'actual_steps':model.num_timesteps,'wall_seconds':time.monotonic()-start,'observation_dim':v.observation_space.shape[0]},indent=2));v.close()
    print('TRAINING COMPLETE',str(folder),flush=True)


def evaluate(mode,seed,episodes,video=False):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    torch.set_num_threads(1)
    params=json.loads((OUT/'base_params.json').read_text())
    folder=OUT/f'{mode}_seed{seed}';folder.mkdir(exist_ok=True,parents=True)
    results=[]
    for ood in (False,True):
        e=RailEnv('base' if mode=='base' else mode,params,ood=ood)
        vec=None;model=None
        if mode!='base':
            vec=VecNormalize.load(folder/'normalization.pkl',DummyVecEnv([lambda:e]));vec.training=False;vec.norm_reward=False
            model=PPO.load(folder/'final',device='cpu')
        for j in range(episodes):
            eval_seed=(20000 if ood else 10000)+j
            obs,_=e.reset(seed=eval_seed)
            total=0
            recorder=None
            if video and j==0:
                from .run import Recorder
                path=folder/('video_ood' if ood else 'video_id');path.mkdir(exist_ok=True)
                recorder=Recorder(e.sim,path,True,subtitle=f'{mode.upper()} controller | rail lift / hold / place | assistance OFF')
            for _ in range(e.max_steps):
                a=np.zeros(26) if model is None else model.predict(vec.normalize_obs(obs[None]),deterministic=True)[0][0]
                obs,r,done,truncated,info=e.step(a);total+=r
                if recorder:recorder.add(f'{mode}: lift hold {e.max_hold:.2f}s / 2s; task success {e.success}')
                if done or truncated:break
            if recorder:recorder.close()
            results.append({'split':'ood' if ood else 'id','eval_seed':eval_seed,'return':total,**info})
        if vec:vec.close()
    report={'controller':mode,'training_seed':seed,'episodes':results,'rates':{split:{'success':float(np.mean([r['success'] for r in results if r['split']==split])),'lift':float(np.mean([r['lift_success'] for r in results if r['split']==split]))} for split in ('id','ood')}}
    (folder/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['rates']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['optimize','train','evaluate'])
    parser.add_argument('--mode',choices=['base','residual','direct'],default='residual')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=131072)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--video',action='store_true')
    a=parser.parse_args()
    if a.operation=='optimize':optimize()
    elif a.operation=='train':train(a.mode,a.seed,a.steps,a.workers)
    else:evaluate(a.mode,a.seed,a.episodes,a.video)
