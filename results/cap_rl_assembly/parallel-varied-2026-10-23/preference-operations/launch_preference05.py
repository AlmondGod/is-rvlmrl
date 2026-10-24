import hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
parent=Path('/opt/cap-speed/varied-run03');out='/opt/cap-speed/varied-preference05'
while not (parent/'selection.json').exists():
 if (parent/'retirement_not_triggered.json').exists():
  print('Contingency not triggered; existing GRPO remains active.',flush=True);raise SystemExit(0)
 time.sleep(2)
assert (parent/'retirement.json').exists(), 'Run was not retired under declared contingency'
assert json.loads((parent/'selection.json').read_text())['retired'] is True
while Path('/proc/8782').exists():
 stat=Path('/proc/8782/stat').read_text().split(') ',1)[1].split()[0]
 if stat=='Z':break
 time.sleep(1)
assert not Path(out).exists()
with open('/opt/cap-speed/varied-preference05.log','w') as log:
 proc=subprocess.Popen(['/opt/cap-speed/train-venv/bin/python','-u','-m','analysis.train_assembly_preferences','--parent',str(parent),'--out',out],cwd='/opt/cap-speed/preference-tools05',stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
receipt=dict(pid=proc.pid,started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent=str(parent),out=out)
Path('/opt/cap-speed/preference-launch05.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
while proc.poll() is None and not (Path(out)/'metadata.json').exists():time.sleep(1)
if not (Path(out)/'metadata.json').exists():raise RuntimeError('Trainer exited before metadata')
toolsdir=Path(out)/'verification_tools';toolsdir.mkdir()
import shutil
sources={}
for source in Path('/opt/cap-speed/preference-tools05/analysis').glob('*.py'):
 shutil.copyfile(source,toolsdir/source.name);sources[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
(toolsdir/'manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
with open('/opt/cap-speed/preference-backup05.log','w') as log:
 backup=subprocess.Popen(['/opt/cap-speed/train-venv/bin/python','-u','/opt/cap-speed/backup_preference_run05.py','--run',out,'--cache','/opt/cap-speed/varied-evidence01','--out','/opt/cap-speed/preference-backups05'],stdout=log,stderr=subprocess.STDOUT)
with open('/opt/cap-speed/preference-verification05.log','w') as log:
 verifier=subprocess.Popen(['/opt/cap-speed/train-venv/bin/python','-u','-m','analysis.verify_varied_assembly','--run',out,'--replay','/opt/cap-speed/preference-replay05','--wait-for-pid',str(proc.pid)],cwd='/opt/cap-speed/preference-tools05',stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
receipt.update(backup_pid=backup.pid,verification_pid=verifier.pid)
Path('/opt/cap-speed/preference-launch05.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(proc.wait())
