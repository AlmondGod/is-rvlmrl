import argparse,hashlib,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

def memory():
 root=Path('/sys/fs/cgroup');stats=dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
 return dict(limit=int((root/'memory.max').read_text()),anon=int(stats['anon']),kernel=int(stats['kernel']),shmem=int(stats['shmem']),current=int((root/'memory.current').read_text()))
def working(m):return m['anon']+m['kernel']+m['shmem']
p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--baseline');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
if a.child:
 Path('/proc/self/oom_score_adj').write_text('1000')
 import cv2
 from experiments.cap_rl import assembly
 from experiments.torso_assembly.acceleration import configure
 from analysis.compare_assembly_speed import compare
 cv2.setNumThreads(1);configure('vectorized_fused')
 base=Path(a.baseline);r=json.loads((base/'result.json').read_text())
 assembly.run(r['seed'],(base/'program.py').read_text(),out,support_profile='contact_and_nails',controller_profile='southwest_approach')
 report=compare(base,out);report['timing_caveat']='One bounded extra simulator overlaps the learner; this is an exactness check, not an isolated throughput measurement.'
 out.with_suffix('.comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 raise SystemExit(0 if report['exact'] else 1)
assert not out.exists();out.mkdir()
initial=memory();assert working(initial)<21_000_000_000 and initial['limit']>=30_999_998_464,initial
receipt=dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),initial_memory=initial,maximum_concurrent_extra_simulators=1,admission_working_limit_bytes=21_000_000_000,stop_working_limit_bytes=25_000_000_000,child_oom_score_adj=1000,scope='Integrated backend exactness only; no extra model or worker pool. Existing learner/frozen sources remain unchanged.',episodes=[])
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
for name in ('success','failure'):
 m=memory()
 if working(m)>=21_000_000_000:raise RuntimeError('Memory admission failed before '+name)
 start=time.monotonic();log=out/(name+'.log');target=out/name
 with log.open('w') as f:
  child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child','--baseline','/opt/cap-speed/native-integrated-'+name+'11','--out',str(target)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 row=dict(name=name,pid=child.pid,started_utc=datetime.now(timezone.utc).isoformat(),peak_working_bytes=working(m))
 receipt['episodes'].append(row)
 reason=None
 while child.poll() is None:
  m=memory();row['peak_working_bytes']=max(row['peak_working_bytes'],working(m))
  if working(m)>=25_000_000_000 or time.monotonic()-start>2400:
   reason='memory_pressure' if working(m)>=25_000_000_000 else 'deadline'
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   break
  (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
  time.sleep(2)
 row.update(exit_code=child.wait(),stop_reason=reason,wall_s=time.monotonic()-start)
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 if reason or row['exit_code']:raise RuntimeError('Preserved unsuccessful infrastructure probe: '+name)
receipt.update(passes=True,completed_utc=datetime.now(timezone.utc).isoformat())
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
