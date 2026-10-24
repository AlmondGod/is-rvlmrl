"""Wait for owned orphan trajectories to finish, then restart without overlap."""
import hashlib,json,os,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/opt/cap-speed');OLD=ROOT/'varied-online13';OUT=ROOT/'varied-online21'
WORKERS=[15563,15566,15567,15568,15569,15570,15571,15572]
AUX=[16157,16666]
def status(pid):
 p=Path(f'/proc/{pid}/stat')
 if not p.exists():return None
 fields=p.read_text().split(') ',1)[1].split()
 return dict(state=fields[0],parent=int(fields[1]),ticks=int(fields[11])+int(fields[12]))
def active(pid):return (s:=status(pid)) is not None and s['state']!='Z'
def incomplete():
 return [p for p in (ROOT/'varied-evidence01').iterdir() if (p/'program.py').exists() and not (p/'certificate.json').exists()]
started=datetime.now(timezone.utc).isoformat();previous={};stable=0
while True:
 current={p:status(p) for p in WORKERS}
 quiet=all(s is None or s['state'] in ('S','Z') for s in current.values())
 unchanged=all(s is None or previous.get(p)==s for p,s in current.items())
 stable=stable+1 if quiet and unchanged and not incomplete() and not any(active(p) for p in AUX) else 0
 if stable>=3:break
 previous=current;time.sleep(5)
for p in WORKERS:
 s=status(p)
 if s and s['state']!='Z':
  if s['parent']!=1:raise RuntimeError('PID is no longer an orphan owned worker')
  command=Path(f'/proc/{p}/cmdline').read_bytes()
  if b'multiprocessing.spawn' not in command and b'spawn_main' not in command:raise RuntimeError('Unexpected worker command')
  os.kill(p,signal.SIGTERM)
# Completed checkpoint backups remain; stop the old now-idle backup waiter.
if active(15384):
 command=Path('/proc/15384/cmdline').read_bytes()
 if b'backup_online_run13.py' not in command:raise RuntimeError('Unexpected old backup PID')
 os.kill(15384,signal.SIGTERM)
while any(active(p) for p in WORKERS):time.sleep(1)
limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
stat=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
working=int(Path('/sys/fs/cgroup/memory.current').read_text())-int(stat.get('inactive_file',0))
if limit-working<24_000_000_000:raise RuntimeError('Insufficient RAM headroom for isolated training restart')
if OUT.exists():raise RuntimeError('Preserve existing recovery output')
with (ROOT/'varied-online21.log').open('w') as log:
 trainer=subprocess.Popen(['/opt/cap-speed/train-venv/bin/python','-u',str(ROOT/'reconstruct_online21.py'),'--parent',str(ROOT/'varied-run03'),'--after-visual-run',str(ROOT/'varied-visual12'),'--out',str(OUT)],cwd=ROOT/'visual-online-tools13',stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
receipt=dict(wait_started_utc=started,started_utc=datetime.now(timezone.utc).isoformat(),pid=trainer.pid,
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),memory_limit_bytes=limit,working_set_before_bytes=working,
 resource_policy='One trainer and eight physical workers; no overlapping experiments or speculative prefetch.')
(ROOT/'online-launch21.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
while trainer.poll() is None and not (OUT/'metadata.json').exists():time.sleep(1)
if not (OUT/'metadata.json').exists():raise RuntimeError('Recovery exited before metadata')
import shutil
toolsdir=OUT/'verification_tools';toolsdir.mkdir();sources={}
for source in (ROOT/'visual-online-tools13/analysis').glob('*.py'):
 shutil.copyfile(source,toolsdir/source.name);sources[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
(toolsdir/'manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
with (ROOT/'online-backup21.log').open('w') as log:
 backup=subprocess.Popen(['/opt/cap-speed/train-venv/bin/python','-u',str(ROOT/'backup_online_run21.py'),'--run',str(OUT),'--cache',str(ROOT/'varied-evidence01'),'--out',str(ROOT/'online-backups21')],stdout=log,stderr=subprocess.STDOUT)
with (ROOT/'online-verification21.log').open('w') as log:
 verifier=subprocess.Popen(['/opt/cap-speed/train-venv/bin/python','-u','-m','analysis.verify_varied_assembly','--run',str(OUT),'--replay',str(ROOT/'online-replay21'),'--wait-for-pid',str(trainer.pid)],cwd=ROOT/'visual-online-tools13',stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
receipt.update(backup_pid=backup.pid,verification_pid=verifier.pid)
(ROOT/'online-launch21.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(trainer.wait())
