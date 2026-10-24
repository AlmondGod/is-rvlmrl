import hashlib,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

def memory():
 root=Path('/sys/fs/cgroup');s=dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
 return dict(limit=int((root/'memory.max').read_text()),working=int(s['anon'])+int(s['kernel'])+int(s['shmem']),current=int((root/'memory.current').read_text()))
prior=Path('/opt/cap-speed/prefix-validation35/admission.json')
while True:
 try:receipt=json.loads(prior.read_text())
 except (FileNotFoundError,json.JSONDecodeError):receipt={}
 if 'exit_code' in receipt:
  assert receipt.get('passes'),receipt
  break
 time.sleep(10)
root=Path('/opt/cap-speed/prefix-validation37');assert not root.exists();root.mkdir()
m=memory();assert m['limit']>=30_999_998_464 and m['working']<21_000_000_000,m
command=[sys.executable,'-u','/opt/cap-speed/prefix37_queue_probe.py']
os.environ['CAP_RL_ASSEMBLY_CACHE']=str(root/'evidence')
def child_priority():Path('/proc/self/oom_score_adj').write_text('1000')
with (root/'episodes.log').open('w') as log:
 child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=child_priority)
started=time.monotonic();receipt=dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pid=child.pid,command=command,initial_memory=m,maximum_concurrent_extra_simulators=1,peak_working_bytes=m['working'],admission_working_limit_bytes=21_000_000_000,stop_working_limit_bytes=25_000_000_000,child_oom_score_adj=1000)
reason=None
while child.poll() is None:
 m=memory();receipt['peak_working_bytes']=max(receipt['peak_working_bytes'],m['working'])
 if m['working']>=25_000_000_000 or time.monotonic()-started>3600:
  reason='memory_pressure' if m['working']>=25_000_000_000 else 'deadline';os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  break
 (root/'admission.json').write_text(json.dumps(receipt,indent=2)+'\n');time.sleep(2)
receipt.update(exit_code=child.wait(),stop_reason=reason,wall_s=time.monotonic()-started,completed_utc=datetime.now(timezone.utc).isoformat())
receipt['passes']=not reason and receipt['exit_code']==0
(root/'admission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
raise SystemExit(0 if receipt['passes'] else 1)
