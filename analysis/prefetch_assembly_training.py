"""Prefetch frozen warm-policy programs for future training layouts only.

This changes execution scheduling, never on-policy sampling or rewards. A result
can enter RL only when the learner independently requests the identical physical
identity through its existing checked cache. All unused work remains explicit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import hashlib,json,multiprocessing,os
from pathlib import Path
import shutil,time


def select_candidates(metadata,report):
    if metadata.get('optimizer_algorithm')!='grpo_visual_balanced_v1':
        raise ValueError('Expected the declared balanced online run')
    if report['weights_sha256']!=metadata['warm_start_weights_sha256'] or report['model_revision']!=metadata['model_revision']:
        raise ValueError('Prefetch candidates must come from this run\'s frozen warm policy')
    allowed=set(metadata['splits']['train'])
    if len(allowed)!=32 or allowed & (set(metadata['splits']['validation'])|set(metadata['splits']['test'])):
        raise ValueError('Prefetch splits must be disjoint')
    if len([r for r in report['rows'] if r['layout_id'] in allowed])!=32:
        raise ValueError('Duplicate or missing training candidates')
    bykey={r['layout_id']:r for r in report['rows'] if r['layout_id'] in allowed}
    if set(bykey)!=allowed or any(not r['exact'] for r in bykey.values()):raise ValueError('Incomplete verified training program coverage')
    first=set(metadata['training_layout_schedule'][0]);keys=[k for group in metadata['training_layout_schedule'] for k in group if k not in first]
    return keys,bykey


def main():
    from experiments.cap_rl import task_assembly as task
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--generation-report',required=True);p.add_argument('--out',required=True);p.add_argument('--trainer-pid',required=True,type=int)
    a=p.parse_args();run=Path(a.run);report_path=Path(a.generation_report);out=Path(a.out)
    metadata=json.loads((run/'metadata.json').read_text());report=json.loads(report_path.read_text())
    keys,bykey=select_candidates(metadata,report)
    task.DATASET_PROFILE=metadata['assembly_split'];task.configure(metadata['support_profile'],metadata['controller_profile'],metadata['assembly_backend'])
    manifest=json.loads((run/'assembly_provenance.json').read_text())
    if task.provenance()!=manifest:raise ValueError('Use the active frozen simulator/runtime')
    task.EXPECTED_PROVENANCE=manifest
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/'prefetch_assembly_training.py');shutil.copyfile(report_path,out/'generation_report.json')
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    info=dict(training_metadata_sha256=hashlib.sha256((run/'metadata.json').read_bytes()).hexdigest(),started_utc=datetime.now(timezone.utc).isoformat(),run=str(run),workers=2,trainer_pid=a.trainer_pid,source_sha256=source_sha,
        generation_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        full_decoder_equivalence=report['passes'],training_candidate_equivalence=True,
        training_layouts=keys,physical_provenance=manifest,cache_root=os.environ['CAP_RL_ASSEMBLY_CACHE'],
        scope='Speculative physical cache fill from frozen warm-policy programs. Training layouts only; no optimizer or reward prediction. RL consumes only independently sampled exact identities; unused evaluations are reported separately.')
    (out/'metadata.json').write_text(json.dumps(info,indent=2)+'\n');started=time.monotonic();count=0
    pending={};remaining=iter(keys)
    with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn'),initializer=task._worker_init,
            initargs=(manifest,metadata['support_profile'],metadata['controller_profile'],metadata['assembly_backend'],1)) as pool:
        def submit_one():
            process=Path(f'/proc/{a.trainer_pid}/stat')
            if (run/'selection.json').exists() or not process.exists():return
            if process.read_text().split(') ',1)[1].split()[0]=='Z':return
            key=next(remaining,None)
            if key is None:return
            layout=task.Layout(int(key.rsplit('-',1)[1]));pending[pool.submit(task.execute,layout,bykey[key]['completion'])]=key
        submit_one();submit_one()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                key=pending.pop(future);result=future.result();row=dict(layout_id=key,completion=bykey[key]['completion'],**result)
                with (out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                count+=1;print(json.dumps(dict(completed=count,layout_id=key,success=result['success'],cache_hit=result.get('cache_hit'),elapsed_seconds=time.monotonic()-started)),flush=True)
                submit_one()
    (out/'completion.json').write_text(json.dumps(dict(completed=count,planned=len(keys),elapsed_seconds=time.monotonic()-started,
        stopped_after_training_selection=(run/'selection.json').exists()),indent=2)+'\n')


if __name__=='__main__':main()
