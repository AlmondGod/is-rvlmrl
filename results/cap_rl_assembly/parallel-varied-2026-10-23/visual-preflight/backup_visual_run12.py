"""Archive immutable checkpoint/evidence increments while the current job runs."""
import argparse, hashlib, io, json, pathlib, tarfile, time
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);a=p.parse_args()
run,cache,out=map(pathlib.Path,(a.run,a.cache,a.out));out.mkdir(exist_ok=False,parents=True)
seen=set();done=set()
def sha(data):return hashlib.sha256(data).hexdigest()
def archive(step,checkpoint):
    name=f'step-{step:04d}';dest=out/(name+'.tar.gz');pending=out/(name+'.partial')
    entries={}
    for folder in ('assembly_source','source','initial_adapter','optimization_source','decoding_source','preference_source','warm_start_adapter','parent_run','observations','constant_prompt_attempt','verification_tools') if step==0 else ():
        for f in (run/folder).rglob('*'):
            if f.is_file():entries['run/'+str(f.relative_to(run))]=f
    for f in run.glob('*.json'):
        if f.name.startswith('validation_') and f.name!='validation_baseline.json':
            try:
                if int(f.stem.split('_')[-1])>step:continue
            except ValueError:continue
        if f.name in ('selection.json','test_comparison.json'):continue
        entries['run/'+f.name]=f
    if checkpoint:
        for f in checkpoint.rglob('*'):
            if f.is_file():entries['run/'+str(f.relative_to(run))]=f
    for f in run.glob('*.jsonl'):
        rows=[json.loads(s) for s in f.read_text().splitlines(keepends=True) if s.endswith('\n') and s.strip()]
        entries['run/'+f.name]=''.join(json.dumps(r)+'\n' for r in rows if r.get('step',0)<=step).encode()
    evidence=[]
    for cert in cache.glob('*/certificate.json'):
        key=cert.parent.name
        if key in seen:continue
        record=json.loads(cert.read_text())
        for filename,digest in record['files'].items():
            f=cert.parent/filename
            assert sha(f.read_bytes())==digest,(key,filename)
            entries['evidence/'+key+'/'+filename]=f
        entries['evidence/'+key+'/certificate.json']=cert;evidence.append(key)
    receipt=dict(format='incremental_training_backup_v1',step=step,previous_steps=sorted(done),
                 run=str(run),cache=str(cache),files_sha256={},new_certified_episodes=evidence,
                 scope='Adapter checkpoint, not optimizer/RNG resume state. Apply increments in order.')
    with tarfile.open(pending,'w:gz',compresslevel=1) as tar:
        for name,value in sorted(entries.items()):
            data=value if isinstance(value,bytes) else value.read_bytes();receipt['files_sha256'][name]=sha(data)
            info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
        data=(json.dumps(receipt,indent=2)+'\n').encode();info=tarfile.TarInfo(f'backup-receipts/step-{step:04d}.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    pending.rename(dest)
    (out/(dest.name+'.sha256')).write_text(sha(dest.read_bytes())+'  '+dest.name+'\n')
    seen.update(evidence);done.add(step);print(json.dumps(dict(step=step,archive=str(dest),new_evidence=len(evidence),bytes=dest.stat().st_size)),flush=True)
while True:
    if (run/'validation_baseline.json').exists() and 0 not in done:archive(0,None)
    for checkpoint in sorted((run/'checkpoints').glob('step-*')):
        if not (checkpoint/'checkpoint.json').exists():continue
        step=int(checkpoint.name.split('-')[-1])
        if step not in done:archive(step,checkpoint)
    if (run/'selection.json').exists():break
    time.sleep(20)
