import pathlib,tarfile,time,os
root=pathlib.Path('/opt/cap-assembly');run=root/'assembly04';out=root/'assembly04-backups';out.mkdir(exist_ok=False)
for step in (1,2,4,8,12,16):
    checkpoint=run/'checkpoints'/f'step-{step:04d}'
    while not (checkpoint/'checkpoint.json').exists():
        if not pathlib.Path('/proc/16142').exists():raise RuntimeError('Trainer stopped before checkpoint '+str(step))
        time.sleep(2)
    partial=out/f'step-{step:04d}.tar.gz.partial';final=out/f'step-{step:04d}.tar.gz'
    with tarfile.open(partial,'w:gz',compresslevel=1) as archive:
        archive.add(checkpoint,arcname=f'step-{step:04d}')
        for name in ('metadata.json','assembly_provenance.json','validation_baseline.json','optimization_source'):
            archive.add(run/name,arcname=name)
    partial.rename(final)
    print(final,final.stat().st_size,flush=True)
