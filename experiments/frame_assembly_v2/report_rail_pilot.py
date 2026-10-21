"""Summarize every frozen-budget run; does not select checkpoints or test cases."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .rail_rl import OUT


def main():
    report=json.loads((OUT/'pilot_results.json').read_text())
    manifest=report['manifest']
    source=Path(__file__).resolve().parent
    matches={n:hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in manifest['source_sha256'].items()}
    matches['base_params.json']=hashlib.sha256((OUT/'base_params.json').read_bytes()).hexdigest()==manifest['base_params_sha256']
    if not all(matches.values()):raise RuntimeError(f'Source changed during the experiment: {matches}')
    rows=[]
    for run in report['runs']:
        for split in ('id','ood'):
            episodes=[e for e in run['episodes'] if e['split']==split]
            rows.append({'run':run['run'],'split':split,'episodes':len(episodes),'lift_successes':sum(e['lift_success'] for e in episodes),'full_task_successes':sum(e['success'] for e in episodes),'posthoc_lift_with_episode_peak_penetration_below_1mm':sum(e['lift_success'] and e['max_penetration_m']<.001 for e in episodes),'mean_max_height_cm':100*np.mean([e['max_lowest_point_m'] for e in episodes]),'mean_max_hold_seconds':np.mean([e['max_hold_seconds'] for e in episodes]),'max_penetration_mm':1000*max(e['max_penetration_m'] for e in episodes)})
    training=[]
    for mode in ('residual','direct'):
        for seed in (0,1):
            folder=OUT/f'{mode}_seed{seed}'
            records=[json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines()]
            training.append({'run':folder.name,'complete':json.loads((folder/'training_complete.json').read_text()),'training_episodes':len(records),'training_lift_successes':sum(bool(e['lift_success']) for e in records),'training_full_task_successes':sum(bool(e['success']) for e in records)})
    summary={'status':'completed fixed-budget exploratory pilot','source_hashes_verified':matches,'total_training_steps':sum(r['complete']['actual_steps'] for r in training),'nominal_search_steps':55000,'training':training,'evaluation':rows,'limitations':['Only two training seeds and 20 paired evaluation cases per split','No real-hardware calibration; ideal wrist carrier','Primary lift metric limits penetration during the hold only. Every scored lift exceeded 1mm earlier or later in its episode; zero pass an additional post-hoc episode-wide 1mm filter. Do not interpret as physically validated grasps.','Original Astra attempts were not rerun as a matched full-task baseline','Direct PPO gets the same nominal-action observation as residual PPO; the ablation is additive control prior','Final placement includes pose, orientation, release and settling; transient airborne contact is insufficient','Archived invalid_reset and superseded_friction directories are excluded from all reported results'],'output_plot':'pilot_results.png'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={'residual':'#1976b8','direct':'#dc7540'}
    for mode in ('residual','direct'):
        for seed in (0,1):
            data=[json.loads(line) for line in (OUT/f'{mode}_seed{seed}/episodes.jsonl').read_text().splitlines()]
            x=np.array([e['step'] for e in data])/1000
            for ax,key,mult in ((axes[0,0],'r',1),(axes[0,1],'max_lowest_point_m',100)):
                values=np.array([e[key] for e in data])*mult
                rolling=np.array([values[max(0,i-19):i+1].mean() for i in range(len(values))])
                ax.plot(x,rolling,color=colors[mode],ls='-' if seed==0 else '--',label=f'{mode}, seed {seed}')
    axes[0,0].set(title='Training return (20-episode mean)',xlabel='Environment steps (thousands)',ylabel='Return');axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='Training: peak whole-rail clearance',xlabel='Environment steps (thousands)',ylabel='Clearance (cm)')
    axes[0,1].axhline(5,color='#777777',ls=':',label='5 cm threshold');axes[0,1].legend(fontsize=8)
    for ax,key,title in ((axes[1,0],'lift_successes','Held-out lift + 2-second stable hold'),(axes[1,1],'full_task_successes','Held-out full lift / move / place / release')):
        for split,offset,color in [('id',-.18,'#1976b8'),('ood',.18,'#b2c9da')]:
            for i,mode in enumerate(('base','residual','direct')):
                subset=[r for r in rows if r['run'].startswith(mode+'_') and r['split']==split]
                rates=np.array([100*r[key]/r['episodes'] for r in subset])
                ax.bar(i+offset,rates.mean(),width=.32,color=color,label=split.upper() if i==0 else None)
                ax.scatter(np.full(len(rates),i+offset),rates,color='#333333',s=18,zorder=3)
                ax.text(i+offset,rates.mean()+3,f'{rates.mean():g}%',ha='center',fontsize=9)
        ax.set(title=title,ylabel='Success (%)',ylim=(0,105),xticks=[0,1,2],xticklabels=['Feedback baseline','Residual PPO','Direct PPO']);ax.legend(fontsize=8)
    fig.suptitle('Wuji rail manipulation — 262,144 training steps per seed\nTwo training seeds; 20 paired evaluation cases per split. Dots show individual policies.',fontsize=13)
    fig.savefig(OUT/'pilot_results.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
