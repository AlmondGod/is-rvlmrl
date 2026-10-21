from pathlib import Path
import argparse,json,gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
parser=argparse.ArgumentParser(); parser.add_argument("root"); root=Path(parser.parse_args().root)
fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for name,label,color in [('delayed_frame_before','Original controller','#ba4646'),('delayed_frame_after','Continuous handoff','#247983')]:
 r=json.loads((root/name/'evaluator/result.json').read_text());t=r['history'][0]['completed_sim_s']
 with gzip.open(root/name/'evaluator/trace.jsonl.gz','rt') as f:
  rows=[x for x in map(json.loads,f) if t-.1<=x['time']<=t+6.1]
 time=np.array([x['time']-t for x in rows]);cmd=np.array([x['command'][0][0] for x in rows]);error=np.array([x['metrics']['frame']['position_error_m'][0] for x in rows])*1000
 ax[0].plot(time,(cmd[:,1]-cmd[0,1])*1000,label=label,color=color)
 ax[1].plot(time,error,label=label,color=color)
for a in ax:
 a.axvline(.02,color='#888',lw=.8,ls=':');a.set_xlabel('Seconds since placement completed');a.grid(alpha=.2);a.spines[['top','right']].set_visible(False)
ax[0].set_ylabel('Commanded Y change (mm)');ax[0].set_title('Target jump removed');ax[0].legend(frameon=False,fontsize=8)
ax[1].set_ylabel('Member 0 placement error (mm)');ax[1].set_title('Less drift while awaiting release')
fig.suptitle('Development test: same layout, six-second supervisor delay',fontsize=12)
fig.savefig(root/'handoff_comparison.png',dpi=160)
