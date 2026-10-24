"""Replay saved perception and independently re-render three frozen test inputs."""
import argparse
from pathlib import Path

from analysis import assembly_visual_prompt as visual
from analysis.assembly_initial_observation import capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();root=Path(args.run);output=Path(args.out)
    if output.exists():raise FileExistsError(output)
    metadata=visual.read(root/'metadata.json');selection=visual.read(root/'selection.json')
    if not visual.enabled(metadata) or selection['best_step']<=0 or selection['validation_success_rate']!=1.:
        raise ValueError('Fresh test input replay requires the frozen learned validation selection')
    source=visual.verify_sources(root,metadata)
    audited=visual.audit_observations(root,metadata,include_test=True)
    destination=root/'initial_observation_replay';destination.mkdir(exist_ok=False)
    rows=[]
    for seed in (53001,53012,53024):
        key=f'assembly-full-v1-{seed}'
        if key not in metadata['splits']['test']:raise ValueError('Fresh input replay split mismatch')
        before=root/'observations'/key;after=destination/key
        capture(seed,after,visual.read(root/'assembly_provenance.json'))
        files=visual.read(before/'receipt.json')['files_sha256']
        matches={name:visual.sha(after/name)==digest for name,digest in files.items()}
        if not all(matches.values()):raise ValueError('Fresh initial sensors differ: '+key)
        rows.append(dict(layout_id=key,exact_files=matches))
    report=dict(passes=True,saved_sensor_replay=audited,fresh_initial_observations=rows,
        source_sha256=visual.sha(__file__),
        scope='All saved visual estimates reproduced; three predeclared held-out reset-time sensor inputs independently re-rendered. No task reward or policy update.')
    visual.dump(output,report)


if __name__=='__main__':main()
