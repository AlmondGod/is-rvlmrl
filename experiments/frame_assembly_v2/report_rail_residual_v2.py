"""Plot the fixed corrected-residual experiment without changing its selection."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .rail_residual_v2 import OUT, STEPS


def main():
    r=json.loads((OUT/'results.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    names=['baseline','legacy_noise','small_noise_start','small_noise_final']
    labels=['No noise','Old noise','New starting\nnoise','New final\nnoise']
    vals=[100*r['exploration'][n]['lift']/r['exploration'][n]['episodes'] for n in names]
    axes[0,0].bar(labels,vals,color=['#777777','#c65d40','#8ab9d1','#247ead'])
    for i,n in enumerate(names):
        c=r['exploration'][n];axes[0,0].text(i,vals[i]+3,f"{c['lift']}/{c['episodes']}",ha='center')
    axes[0,0].set(title='Exploration alone: no policy learning',ylabel='Stable-lift success (%)',ylim=(0,105))
    for seed,color in [(0,'#247ead'),(1,'#c65d40')]:
        data=[json.loads(x) for x in (OUT/f'seed{seed}/episodes.jsonl').read_text().splitlines()]
        values=np.array([e['lift_success'] for e in data])*100
        mean=[values[max(0,i-29):i+1].mean() for i in range(len(values))]
        axes[0,1].plot([e['step']/1000 for e in data],mean,color=color,label=f'seed {seed}')
    axes[0,1].set(title='Training with bounded, sampled residuals',xlabel='Environment steps (thousands)',ylabel='Stable lift (30-episode mean, %)',ylim=(0,105));axes[0,1].legend()
    for ax,selected,title in [(axes[1,0],False,'Final learned policies on fresh test cases'),(axes[1,1],True,'Validation-selected controller on fresh test cases')]:
        candidates=['baseline']+[r['selected'][f'seed{s}'] if selected else f'seed{s}/step{STEPS}' for s in (0,1)]
        for split,offset,color in [('id',-.18,'#247ead'),('ood',.18,'#a6c7d8')]:
            for i,label in enumerate(candidates):
                c=r['test'][label][split];rate=100*c['lift']/c['episodes']
                ax.bar(i+offset,rate,width=.32,color=color,label={'id':'Standard','ood':'Harder'}[split] if i==0 else None)
                ax.text(i+offset,rate+3,f"{c['lift']}/{c['episodes']}",ha='center',fontsize=9)
        ticklabels=['Baseline']+[f'Seed {s}'+('\n(baseline retained)' if selected and candidates[s+1]=='baseline' else '') for s in (0,1)]
        ax.set(title=title,ylabel='Stable-lift success (%)',ylim=(0,105),xticks=range(3),xticklabels=ticklabels);ax.legend(fontsize=8)
    fig.suptitle('Exact-zero initialization + corrections capped at 10%\nExploratory simulation results; unresolved contact penetration limits physical interpretation',fontsize=13)
    fig.savefig(OUT/'results.png',dpi=160);plt.close(fig)
    lines=['# Corrected residual RL pilot','',
           'The physical environment and reward were held fixed. Both networks start with exactly zero deterministic output. Corrections are capped at 10% of each action dimension; effective exploration standard deviation ramps from 0.005 to 0.03 over 65,536 steps. Learning rate is 0.0001 and entropy bonus is zero. This package changes several training choices; the noise-only comparison separately isolates exploration.', '',
           'Two seeds each trained for 262,144 steps. Validation used 20 standard cases (30000–30019). Checkpoints at 131,072 and 262,144 steps competed with the zero-residual baseline. Selection prioritizes full-task successes, then stable lifts; ties retain the baseline or earlier checkpoint. Fresh tests use 20 standard cases (40000–40019) and 20 harder cases (50000–50019). Test outcomes were unavailable when selection was saved.','',
           '## Initialization and exploration','',
           'Three complete episodes had exactly identical simulator positions and velocities with the zero-output policy and baseline. The network also passed an arbitrary-observation zero-output test.','',
           '| No-learning condition | Stable lift | Full task |','|---|---:|---:|']
    for label,c in r['exploration'].items():lines.append(f"| {label} | {c['lift']}/{c['episodes']} | {c['full_success']}/{c['episodes']} |")
    lines+=['','## Validation','', '| Candidate | Stable lift | Full task |','|---|---:|---:|']
    for label,c in r['validation'].items():lines.append(f"| {label} | {c['lift']}/{c['episodes']} | {c['full_success']}/{c['episodes']} |")
    lines+=['',f"Selected controllers: {json.dumps(r['selected'])}",'','## Fresh test results','', '| Controller | Standard lift | Harder lift | Full task (40 cases) | Episode-wide <1 mm lift (40 cases) |','|---|---:|---:|---:|---:|']
    for label,splits in r['test'].items():
        a,b=splits['id'],splits['ood']
        lines.append(f"| {label} | {a['lift']}/20 | {b['lift']}/20 | {a['full_success']+b['full_success']}/40 | {a['strict_episode_lift']+b['strict_episode_lift']}/40 |")
    lines+=['','## Interpretation limits','',
            'The primary lift criterion rejects penetration of 1 mm or more during the qualifying two-second hold. It does not reject penetration elsewhere in the episode. The stricter episode-wide column is a secondary diagnostic, not the predeclared controller-selection metric. The uncalibrated simulation, ideal wrist carrier, small validation set and two training seeds do not establish real-world grasp performance. A validation fallback is not a guarantee against regression on new cases.', '',
            f"Total elapsed experiment time: {r['wall_seconds']/60:.1f} minutes. All recorded source/controller hashes were verified: {all(r['frozen_hashes_verified'].values())}.", '',
            'Artifacts: manifest.json, initialization checks, exploration_only.json, per-seed models/normalizers/training logs, validation.json, selection.json, test.json, results.json and results.png. Original v1 artifacts remain in ../rl/.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(OUT/'report.md')


if __name__=='__main__':main()
