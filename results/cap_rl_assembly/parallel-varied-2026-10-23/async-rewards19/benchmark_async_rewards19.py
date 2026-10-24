import json,os,time
from pathlib import Path
from experiments.cap_rl import task_assembly as task
from analysis.compare_assembly_speed import compare
from assembly_async_rewards19 import assembly_queue


def main():
 root=Path('/opt/cap-speed/async-reward19');root.mkdir(exist_ok=False)
 task.configure('contact_and_nails','southwest_approach','vectorized');task.PHYSICS_WORKERS=8;task.PHYSICS_THREADS=1
 task.EXPECTED_PROVENANCE=json.loads(Path('/opt/cap-speed/varied-online13/assembly_provenance.json').read_text())
 if task.provenance()!=task.EXPECTED_PROVENANCE:raise RuntimeError('Frozen physical source mismatch')
 groups=[([task.Layout(s)]*2,['stop()','finish()']) for s in range(51001,51005)]
 records={}
 for name in ('coordinator','independent'):
  os.environ['CAP_RL_ASSEMBLY_CACHE']=str(root/name)
  start=time.monotonic()
  if name=='coordinator':
   pending=[task.submit_batch(*group) for group in groups]
   rows=[p.result() for p in pending];task.shutdown_workers()
  else:
   queue=assembly_queue(task,workers=8)
   pending=[queue.submit(*group) for group in groups]
   rows=[p.result() for p in pending];queue.executor.shutdown()
  records[name]=dict(elapsed_seconds=time.monotonic()-start,rows=rows)
  (root/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
 comparisons=[]
 for a,b in zip(records['coordinator']['rows'],records['independent']['rows']):
  for x,y in zip(a,b):comparisons.append(compare(x['evidence'],y['evidence']))
 report=dict(passes=all(c['exact'] for c in comparisons),records=records,comparisons=comparisons,
  speedup=records['coordinator']['elapsed_seconds']/records['independent']['elapsed_seconds'],
  scope='Eight real early-terminal 6-second physics episodes in four groups, not complete assemblies or RL improvement. Separate caches; shared host; includes pool startup/shutdown.')
 (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('records','comparisons')}),flush=True)
 if not report['passes']:raise SystemExit(1)


if __name__=='__main__':main()
