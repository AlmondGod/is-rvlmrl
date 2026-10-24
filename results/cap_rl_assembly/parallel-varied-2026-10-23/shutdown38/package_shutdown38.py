from pathlib import Path
import hashlib,json,tarfile,io,os,signal
root=Path('/opt/cap-speed');run=root/'varied-online25';out=root/'shutdown38';files={}
try:os.kill(18160,signal.SIGTERM)
except ProcessLookupError:pass
cutoff=(run/'checkpoints/step-0004/checkpoint.json').stat().st_mtime
for p in run.iterdir():
 if p.is_file():files['run/'+p.name]=p
for folder in ('online_retirement_source',):
 for p in (run/folder).rglob('*'):
  if p.is_file():files['run/'+str(p.relative_to(run))]=p
new_evidence=[]
for d in (root/'varied-evidence01').iterdir():
 if not d.is_dir() or d.name.startswith('.'):continue
 payloads=[p for p in d.iterdir() if p.is_file()]
 if not payloads or max(p.stat().st_mtime for p in payloads)<cutoff:continue
 new_evidence.append(d.name)
 for p in payloads:files['evidence/'+d.name+'/'+p.name]=p
for folder in ('prefix-validation35','prefix-tools35','prefix-tools36','shutdown38'):
 for p in (root/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in str(p):files[folder+'/'+str(p.relative_to(root/folder))]=p
for name in ('varied-online25.log','online-retirement34.log','online-verification25.log','online-backup25.log','online-launch25.json','varied-gpu-telemetry.csv','prefix35-tests.log','prefix36-spawn.log','prefix-validation35.log','prefix-validation37.log','prefix37_queue_probe.py','guard_prefix37.py','guard_prefix35.py','shutdown_varied38.py','package_shutdown38.py'):
 if (root/name).is_file():files['tools/'+name]=root/name
receipt=dict(format='user_shutdown_increment_v1',requires='online25 step-0000 through step-0004 archives, already verified on GitHub',new_or_partial_evidence=new_evidence,files_sha256={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sorted(files.items())},scope='User-requested shutdown. Four complete rounds and sixteen updates. Validation 4 interrupted; no selection or held-out result. Prefix35 partial, prefix37 never started. Interrupted episodes are not certified rewards.')
archive=root/'cap-rl-shutdown-2026-10-24.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for name,p in files.items():tar.add(p,arcname=name,recursive=False)
 data=(json.dumps(receipt,indent=2)+'\n').encode();info=tarfile.TarInfo('manifest.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
summary=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files),new_or_partial_episodes=len(new_evidence))
(out/'archive.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
