"""Preserve historical physical evidence without including any held-out result."""
import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path('/opt/cap-assembly')
sys.path.insert(0,str(root/'verification-tools-v5'))
from cap_rl_execution_inventory import inventory
roots=[root/n for n in ['assembly01-evidence','assembly02-evidence',
 'assembly02-permutation-prefetch-evidence','assembly02-speculative-evidence',
 'assembly02-speculative-evidence-v2','assembly02-validation-prefetch-evidence',
 'assembly03-evidence','parallel-check','corrected-parallel-check',
 'feasibility-train41001','support-v2-train41004']]
directories=set()
for source in roots:
 for p in source.rglob('program.py'):
  directory=p.parent
  if not (directory/'provenance.json').exists():continue
  result=directory/'result.json'
  if result.exists():
   if 43001<=json.loads(result.read_text())['seed']<=43008:continue
  elif source.name=='assembly03-evidence':continue
  directories.add(directory)
valid=[];integrity_errors=[]
for directory in sorted(directories):
 try:inventory([directory]);valid.append(directory)
 except ValueError as error:integrity_errors.append(dict(directory=str(directory),error=str(error)))
report=inventory(valid)
report['excluded_integrity_errors']=integrity_errors
report['scope']+=' Historical training/validation/development only; held-out test seeds excluded. Explicitly retains incomplete and intentionally tampered diagnostic artifacts.'
files=set()
for directory in directories:files.update(p for p in directory.rglob('*') if p.is_file())
for name in ['assembly01','assembly02','assembly03']:
 run=root/name
 files.update(p for p in run.iterdir() if p.is_file())
 if (run/'source').exists():files.update((run/'source').glob('*.py'))
 if (run/'checkpoints').exists():files.update((run/'checkpoints').glob('*/checkpoint.json'))
for name in ['corrected-controller-verification.json','corrected-parallel-check.log',
 'assembly03-training-prefetch.json','assembly03-validation-prefetch.json',
 'assembly02-permutation-prefetch.json','retire_assembly03_if_stalled.py']:
 if (root/name).is_file():files.add(root/name)
manifest=dict(scope=report['scope'],files=[],driver_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
out=root/'assembly-exploration-evidence.tar.gz';partial=out.with_suffix(out.suffix+'.partial')
assert not out.exists() and not partial.exists()
with tarfile.open(partial,'w:gz',compresslevel=1) as archive:
 for source in sorted(files):
  assert not source.is_symlink()
  member=str(source.relative_to(root));digest=hashlib.sha256(source.read_bytes()).hexdigest()
  archive.add(source,arcname=member,recursive=False)
  manifest['files'].append(dict(member=member,sha256=digest))
 for name,value in [('exploration_inventory.json',report),('exploration_manifest.json',manifest)]:
  payload=(json.dumps(value,indent=2)+'\n').encode();info=tarfile.TarInfo(name);info.size=len(payload)
  archive.addfile(info,io.BytesIO(payload))
partial.rename(out)
(root/'assembly-exploration-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
 physical_executions=report['completed_physical_executions'],integrity_errors=integrity_errors)),flush=True)
