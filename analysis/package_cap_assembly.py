"""Package a verified run, its selected weights and every referenced trajectory.

Records retain their original paths and bytes. bundle_manifest.json maps those
paths to archive members; it never rewrites a physical evidence certificate.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--replay', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    run, replay, out = map(Path, (args.run, args.replay, args.out))
    for report in (run/'record_audit.json', run/'generation_audit.json', replay/'audit.json'):
        if not json.loads(report.read_text())['passes']:
            raise ValueError('Verification did not pass: '+str(report))
    selection = json.loads((run/'selection.json').read_text())
    checkpoint = run/'checkpoints'/f"step-{selection['best_step']:04d}"
    if not (checkpoint/'checkpoint.json').is_file():
        raise ValueError('Selected trained checkpoint is absent')
    rows = [json.loads(s) for s in (run/'rollouts.jsonl').read_text().splitlines()]
    for path in [run/'validation_baseline.json', *sorted(run.glob('validation_[0-9]*.json'))]:
        rows.extend(json.loads(path.read_text())['rows'])
    test = json.loads((run/'test_comparison.json').read_text())
    rows.extend(test['before']+test['after'])
    evidence = {r['evidence_id']: Path(r['evidence']) for r in rows if r.get('evidence_id')}
    sources = [(p, Path('run')/p.relative_to(run)) for p in run.rglob('*')
               if p.is_file() and 'last_adapter' not in p.relative_to(run).parts
               and ('checkpoints' not in p.relative_to(run).parts or checkpoint in p.parents
                    or p.name == 'checkpoint.json')]
    for key, directory in evidence.items():
        if directory.name != key or not (directory/'certificate.json').is_file():
            raise ValueError('Missing certified evidence: '+str(directory))
        sources.extend((p, Path('evidence')/key/p.relative_to(directory))
                       for p in directory.rglob('*') if p.is_file())
    sources.extend((p, Path('independent_replay')/p.relative_to(replay))
                   for p in replay.rglob('*') if p.is_file())
    if out.exists() or out.with_suffix(out.suffix+'.partial').exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(selected_step=selection['best_step'],
                    evidence_root_mapping={str(v):'evidence/'+k for k,v in evidence.items()},
                    files=[])
    partial = out.with_suffix(out.suffix+'.partial')
    with tarfile.open(partial, 'w:gz', compresslevel=1) as archive:
        for source, member in sorted(sources):
            if source.is_symlink():
                raise ValueError('Unexpected evidence symlink: '+str(source))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            archive.add(source, arcname=str(member), recursive=False)
            manifest['files'].append(dict(member=str(member), source=str(source), sha256=digest))
        data = (json.dumps(manifest, indent=2)+'\n').encode()
        info = tarfile.TarInfo('bundle_manifest.json'); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    partial.rename(out)
    print(json.dumps(dict(path=str(out), bytes=out.stat().st_size,
                          sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                          files=len(sources), certified_trajectories=len(evidence))), flush=True)


if __name__ == '__main__':
    main()
