import pathlib,tarfile,time,os
root=pathlib.Path('/opt/cap-assembly');run=root/'assembly03';out=root/'assembly03-backups';out.mkdir(exist_ok=False)
for step in (1,8,16,24,32):
    checkpoint=run/'checkpoints'/f'step-{step:04d}'
    while not (checkpoint/'checkpoint.json').exists():
        if not pathlib.Path('/proc/13729').exists():raise RuntimeError('Trainer stopped before checkpoint '+str(step))
        time.sleep(2)
    partial=out/f'step-{step:04d}.tar.gz.partial';final=out/f'step-{step:04d}.tar.gz'
    with tarfile.open(partial,'w:gz',compresslevel=1) as archive:
        archive.add(checkpoint,arcname=f'step-{step:04d}')
        for name in ('metadata.json','assembly_provenance.json','validation_baseline.json'):
            archive.add(run/name,arcname=name)
    partial.rename(final)
    print(final,final.stat().st_size,flush=True)
