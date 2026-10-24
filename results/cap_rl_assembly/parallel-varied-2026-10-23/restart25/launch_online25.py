import hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
parent=Path('/opt/cap-speed/varied-run03');out='/opt/cap-speed/varied-online25'
assert json.loads(Path('/opt/cap-speed/varied-online21/reconstruction_verification.json').read_text())['passes'] is False
assert not Path('/proc/17779').exists(), 'Previous trainer must be stopped'
limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
stats=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
working=int(Path('/sys/fs/cgroup/memory.current').read_text())-int(stats.get('inactive_file',0))
assert limit-working>=24_000_000_000, 'Insufficient isolated training RAM'
assert not Path(out).exists()
with open('/opt/cap-speed/varied-online25.log','w') as log:
 proc=subprocess.Popen(['/opt/cap-speed/train-venv/bin/python','-u','-m','analysis.train_assembly_visual_grpo','--parent',str(parent),'--out',out,'--after-visual-run','/opt/cap-speed/varied-visual12','--prior-interrupted-run','/opt/cap-speed/varied-online13','--prior-reconstruction-run','/opt/cap-speed/varied-online21'],cwd='/opt/cap-speed/visual-online-tools25',stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
receipt=dict(pid=proc.pid,started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent=str(parent),out=out)
Path('/opt/cap-speed/online-launch25.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
while proc.poll() is None and not (Path(out)/'metadata.json').exists():time.sleep(1)
if not (Path(out)/'metadata.json').exists():raise RuntimeError('Trainer exited before metadata')
toolsdir=Path(out)/'verification_tools';toolsdir.mkdir()
import shutil
sources={}
for source in Path('/opt/cap-speed/visual-online-tools25/analysis').glob('*.py'):
 shutil.copyfile(source,toolsdir/source.name);sources[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
(toolsdir/'manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
with open('/opt/cap-speed/online-backup25.log','w') as log:
 backup=subprocess.Popen(['/opt/cap-speed/train-venv/bin/python','-u','/opt/cap-speed/backup_online_run25.py','--run',out,'--cache','/opt/cap-speed/varied-evidence01','--out','/opt/cap-speed/online-backups25'],stdout=log,stderr=subprocess.STDOUT)
with open('/opt/cap-speed/online-verification25.log','w') as log:
 verifier=subprocess.Popen(['/opt/cap-speed/train-venv/bin/python','-u','-m','analysis.verify_varied_assembly','--run',out,'--replay','/opt/cap-speed/online-replay25','--wait-for-pid',str(proc.pid)],cwd='/opt/cap-speed/visual-online-tools25',stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
receipt.update(backup_pid=backup.pid,verification_pid=verifier.pid)
Path('/opt/cap-speed/online-launch25.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(proc.wait())
