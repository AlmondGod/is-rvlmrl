"""Predeclared no-improvement cutoff for the fourth balanced GRPO round.

This external supervisor never edits the frozen learner or its physics. It can
stop only at the completed validation boundary, preserves the fourth complete
checkpoint and writes an explicit retirement rather than claiming a ceiling.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    path=Path(path);temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def decision(metadata,baseline,reports):
    if metadata.get('optimizer_algorithm')!='grpo_visual_balanced_v1' or metadata['steps']!=8:
        raise ValueError('Unexpected online protocol')
    keys=metadata['splits']['validation']
    if len(keys)!=8 or len(set(keys))!=8 or set(reports)!={2,4}:
        raise ValueError('Expected both scheduled eight-layout validations')
    for report in [baseline,*reports.values()]:
        rows=report['rows'];metrics=report['metrics']
        if ([r['layout_id'] for r in rows]!=keys or metrics['n']!=8 or
            metrics['successes']!=sum(bool(r['success']) for r in rows) or
            metrics['success_rate']!=metrics['successes']/8):
            raise ValueError('Validation coverage or aggregate mismatch')
    if baseline['metrics']['successes']!=4:raise ValueError('Expected the frozen 4/8 baseline')
    return max(r['metrics']['successes'] for r in reports.values())<=4


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--pid',required=True,type=int)
    parser.add_argument('--cache',required=True);args=parser.parse_args()
    root=Path(args.run).resolve();cache=Path(args.cache).resolve()
    if (root/'validation_0004.json').exists():raise ValueError('Declare this contingency before validation 4')
    metadata=read(root/'metadata.json')
    proc=Path(f'/proc/{args.pid}')
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(root) not in command or 'analysis.train_assembly_visual_grpo' not in command:
        raise ValueError('PID is not the intended frozen learner')
    start_ticks=(proc/'stat').read_text().split()[21]
    sources=root/'online_retirement_source';sources.mkdir(exist_ok=False)
    for name in ('retire_assembly_online.py','audit_assembly_online_retirement.py'):
        shutil.copyfile(Path(__file__).with_name(name),sources/name)
    policy=dict(format='online_no_improvement_policy_v1',declared_utc=utc(),boundary=4,
        condition='Both scheduled validation checkpoints fail to exceed the initial 4/8 strict-success score.',
        rationale='Retire the small-layout-batch configuration if still unimproved; prepare full-layout cohort updates and shared-prefix acceleration using training/validation evidence only.',
        fallback_scope='No held-out outcomes, source changes, reward relaxation, or exact-resume claim. Any new phase freezes its own policy/rollout/selection protocol before execution.',
        learner_pid=args.pid,learner_start_ticks=start_ticks,metadata_sha256=sha(root/'metadata.json'),
        sources_sha256={p.name:sha(p) for p in sources.iterdir()})
    write(root/'online_retirement_policy.json',policy)
    print(json.dumps(dict(stage='declared',**policy)),flush=True)
    while proc.exists() and not (root/'validation_0004.json').exists():time.sleep(.2)
    if not proc.exists():raise RuntimeError('Learner exited before the decision boundary')
    for _ in range(100):
        try:reports={step:read(root/f'validation_{step:04d}.json') for step in (2,4)};break
        except json.JSONDecodeError:time.sleep(.02)
    else:raise RuntimeError('Fourth validation did not finish writing')
    baseline=read(root/'validation_baseline.json')
    if not decision(metadata,baseline,reports):
        write(root/'online_retirement_decision.json',dict(retired=False,utc=utc(),reason='Validation improved; original frozen protocol continues.'))
        return
    if (proc/'stat').read_text().split()[21]!=start_ticks:raise RuntimeError('Learner PID changed')
    os.kill(args.pid,signal.SIGSTOP)
    stopped=False
    try:
        for _ in range(100):
            if (proc/'stat').read_text().split()[2]=='T':stopped=True;break
            time.sleep(.02)
        if not stopped:raise RuntimeError('Learner did not stop at boundary')
        training=[json.loads(s) for s in (root/'training.jsonl').read_text().splitlines()]
        inflight=[str(p) for p in cache.glob('*/program.py') if not (p.parent/'result.json').exists()]
        if [r['step'] for r in training]!=[1,2,3,4] or inflight:
            write(root/'online_retirement_decision.json',dict(retired=False,utc=utc(),reason='Physical boundary already passed; original protocol continues.',inflight=inflight))
            os.kill(args.pid,signal.SIGCONT);stopped=False
            return
        checkpoint=root/'checkpoints/step-0004';cert=read(checkpoint/'checkpoint.json')
        for name,digest in cert['files_sha256'].items():
            if sha(checkpoint/name)!=digest:raise RuntimeError('Fourth checkpoint payload changed')
        if (root/'selection.json').exists():raise RuntimeError('Selection already exists')
        os.kill(args.pid,signal.SIGINT);os.kill(args.pid,signal.SIGCONT);stopped=False
        for _ in range(600):
            if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
            time.sleep(.1)
        else:raise RuntimeError('Learner did not exit; preserve boundary for inspection')
        shutil.copytree(checkpoint,root/'last_adapter',ignore=shutil.ignore_patterns('training_state.pt','checkpoint.json'))
        started=datetime.fromisoformat(read(root/'process_start.json')['started_utc'])
        selected=dict(best_step=0,validation_success_rate=.5,validation_mean_reward=.5,
            selection_split='validation',training_seconds=(datetime.now(timezone.utc)-started).total_seconds(),
            completed_rollout_rounds=4,completed_optimizer_updates=16,planned_rollout_rounds=8,
            stopped_at_validation_ceiling=False,retired=True,
            timing_scope='Full process wall time through retirement, including setup/baseline and possible partial next-generation work; not just optimizer time.')
        names=['metadata.json','training.jsonl','rollouts.jsonl','validation_baseline.json',
               'validation_0002.json','validation_0004.json','checkpoints/step-0004/checkpoint.json',
               'best_adapter/adapter_model.safetensors','last_adapter/adapter_model.safetensors']
        receipt=dict(format='online_no_improvement_retirement_v1',completed_rounds=4,optimizer_updates=16,
            test_evaluated=False,policy_sha256=sha(root/'online_retirement_policy.json'),
            validation_completed_utc=datetime.fromtimestamp((root/'validation_0004.json').stat().st_mtime,timezone.utc).isoformat(),
            retired_utc=utc(),partial_next_generation_may_have_started=True,
            files_sha256={name:sha(root/name) for name in names})
        write(root/'online_retirement.json',receipt)
        write(root/'selection.json',selected)
        print(json.dumps(dict(stage='retired',selection=selected,receipt=receipt)),flush=True)
    finally:
        if stopped and proc.exists():os.kill(args.pid,signal.SIGCONT)


if __name__=='__main__':main()
