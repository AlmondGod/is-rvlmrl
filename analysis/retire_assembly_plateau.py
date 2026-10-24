"""Retire the declared GRPO pilot at six rounds only if greedy validation is unchanged."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import time

from analysis.finalize_assembly_ceiling import read, running, save, sha


def same_policy_results(a,b):
    return (a['metrics']==b['metrics'] and
            [(r['layout_id'],r['completion'],r['success'],r['reward']) for r in a['rows']]==
            [(r['layout_id'],r['completion'],r['success'],r['reward']) for r in b['rows']])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--pid',required=True,type=int)
    p.add_argument('--protocol',required=True)
    a=p.parse_args();root=Path(a.run).resolve();metadata=read(root/'metadata.json')
    assert metadata['steps']==16 and metadata['eval_every']==2
    policy=root/'retirement_policy.json'
    if policy.exists() or (root/'validation_0006.json').exists():raise ValueError('Register before round-six validation')
    source=root/'retirement_source';source.mkdir()
    shutil.copyfile(__file__,source/Path(__file__).name)
    shutil.copyfile(a.protocol,source/'protocol.md')
    save(policy,dict(format='assembly_plateau_retirement_policy_v1',registered_utc=datetime.now(timezone.utc).isoformat(),
        boundary=6,condition='All scheduled greedy validation programs and strict outcomes unchanged from baseline',
        source_sha256=sha(__file__),protocol_sha256=sha(a.protocol)))
    target=root/'validation_0006.json'
    while running(a.pid):
        try:
            report=read(target)
            break
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(.2)
    else:raise RuntimeError('Training stopped before declared plateau boundary')
    baseline=read(root/'validation_baseline.json')
    if not all(same_policy_results(baseline,read(root/f'validation_{s:04d}.json')) for s in (2,4,6)):
        save(root/'retirement_not_triggered.json',dict(reason='Validation changed; continue original GRPO budget'))
        return
    if any((root/name).exists() for name in ('selection.json','test_comparison.json','raw_test_baseline.json')):
        raise ValueError('Run already selected or tested')
    command=Path(f'/proc/{a.pid}/cmdline').read_bytes()
    if str(root).encode() not in command or b'analysis.cap_rl_assembly_grammar' not in command:
        raise ValueError('Unexpected training process')
    os.kill(a.pid,signal.SIGINT)
    while running(a.pid):time.sleep(1)
    updates=[json.loads(s) for s in (root/'training.jsonl').read_text().splitlines()]
    assert [r['step'] for r in updates]==list(range(1,7)), 'Advanced beyond retirement boundary'
    checkpoint=root/'checkpoints/step-0006';cert=read(checkpoint/'checkpoint.json')
    last=root/'last_adapter';last.mkdir()
    for name,digest in cert['files_sha256'].items():
        assert sha(checkpoint/name)==digest;shutil.copyfile(checkpoint/name,last/name)
    assert sha(root/'best_adapter/adapter_model.safetensors')==sha(root/'initial_adapter/adapter_model.safetensors')
    names=['metadata.json','training.jsonl','rollouts.jsonl','validation_baseline.json','retirement_policy.json']
    names += [f'validation_{s:04d}.json' for s in (2,4,6)]
    save(root/'retirement.json',dict(format='assembly_plateau_retirement_v1',completed_rounds=6,
        planned_rounds=16,optimizer_updates=24,test_evaluated=False,
        reason='Unchanged scheduled greedy validation; reallocate remaining development budget to certified preference replay',
        files_sha256={n:sha(root/n) for n in names},source_sha256=sha(__file__)))
    elapsed=updates[-1]['elapsed_seconds']+max(0.,target.stat().st_mtime-(root/'training.jsonl').stat().st_mtime)
    save(root/'selection.json',dict(best_step=0,validation_success_rate=baseline['metrics']['success_rate'],
        validation_mean_reward=baseline['metrics']['mean_reward'],selection_split='validation',training_seconds=elapsed,
        retired=True,timing_scope='Loop elapsed plus final validation/checkpoint timestamp interval'))
    print(json.dumps(dict(retired=True,rounds=6,optimizer_updates=24,test_evaluated=False)),flush=True)


if __name__=='__main__':main()
