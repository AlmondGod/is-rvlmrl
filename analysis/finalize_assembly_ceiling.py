"""Stop only at a scheduled perfect-validation boundary, before any test use.

Preserves the originally planned budget and records a source-hashed stopping
receipt. It never chooses an order, changes weights, or reads held-out outcomes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    p=Path(path);temporary=p.with_suffix(p.suffix+'.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(p)
def running(pid):
    p=Path(f'/proc/{pid}/stat')
    try:return p.read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def finalize(root,pid,step):
    metadata=read(root/'metadata.json')
    if (root/'test_comparison.json').exists() or (root/'raw_test_baseline.json').exists():
        raise ValueError('Held-out evaluation must not precede stopping')
    if not 0<step<metadata['steps'] or step%metadata['eval_every']:
        raise ValueError('Not an early scheduled validation boundary')
    report=read(root/f'validation_{step:04d}.json')
    assert report['metrics']['success_rate']==report['metrics']['mean_reward']==1.
    assert all(r['success'] and r['reward']==1. for r in report['rows'])
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
    if str(root).encode() not in command or b'analysis.cap_rl_assembly_' not in command:
        raise ValueError('PID is not the requested assembly training run')
    os.kill(pid,signal.SIGINT)
    while running(pid):time.sleep(1)
    updates=[json.loads(s) for s in (root/'training.jsonl').read_text().splitlines()]
    assert [r['step'] for r in updates]==list(range(1,step+1)), 'Training advanced past stopping boundary'
    validation=[(0,read(root/'validation_baseline.json'))]
    validation += [(s,read(root/f'validation_{s:04d}.json')) for s in range(metadata['eval_every'],step+1,metadata['eval_every'])]
    best_step,best=max(validation,key=lambda pair:(pair[1]['metrics']['success_rate'],pair[1]['metrics']['mean_reward']))
    def copy_checkpoint(number,destination):
        source=root/'checkpoints'/f'step-{number:04d}'
        cert=read(source/'checkpoint.json')
        destination.mkdir(exist_ok=True)
        for name,digest in cert['files_sha256'].items():
            assert sha(source/name)==digest
            pending=destination/(name+'.pending');shutil.copyfile(source/name,pending);pending.replace(destination/name)
    if best_step:copy_checkpoint(best_step,root/'best_adapter')
    else:
        assert sha(root/'best_adapter/adapter_model.safetensors')==sha(root/'initial_adapter/adapter_model.safetensors')
    copy_checkpoint(step,root/'last_adapter')
    names=['metadata.json','training.jsonl','rollouts.jsonl','validation_ceiling_policy.json','validation_baseline.json']
    names += [f'validation_{s:04d}.json' for s in range(metadata['eval_every'],step+1,metadata['eval_every'])]
    source=root/'selection_source'/Path(__file__).name;source.parent.mkdir(exist_ok=True)
    shutil.copyfile(__file__,source)
    save(root/'validation_ceiling_stop.json',dict(format='cap_rl_validation_ceiling_stop_v2',
        upper_bound=[1.,1.],test_evaluated=False,completed_rollout_rounds=step,
        planned_rollout_rounds=metadata['steps'],optimizer_updates=step*metadata.get('update_epochs',1),
        selected_step=best_step,source_path=str(source.relative_to(root)),source_sha256=sha(source),
        files_sha256={name:sha(root/name) for name in names}))
    elapsed=updates[-1]['elapsed_seconds']+max(0.,(root/f'validation_{step:04d}.json').stat().st_mtime-(root/'training.jsonl').stat().st_mtime)
    save(root/'selection.json',dict(best_step=best_step,validation_success_rate=best['metrics']['success_rate'],
        validation_mean_reward=best['metrics']['mean_reward'],selection_split='validation',training_seconds=elapsed,
        timing_scope='Last rollout elapsed time plus final validation/checkpoint wall time from file timestamps'))
    print(json.dumps(dict(stopped_at_validation_ceiling=True,rollout_rounds=step,selected_step=best_step)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--pid',required=True,type=int)
    args=p.parse_args();root=Path(args.run).resolve()
    while not (root/'metadata.json').exists():
        if not running(args.pid):raise RuntimeError('Training exited before metadata')
        time.sleep(1)
    while True:
        try:metadata=read(root/'metadata.json');break
        except json.JSONDecodeError:time.sleep(1)
    if metadata.get('optimization_reward')!='strict_success':raise ValueError('Binary strict reward required')
    policy=root/'validation_ceiling_policy.json'
    if policy.exists():raise FileExistsError(policy)
    if (root/'training.jsonl').exists():raise ValueError('Register stopping policy before the first completed round')
    save(policy,dict(upper_bound=[1.,1.],minimum_rounds=metadata['eval_every'],
        selection_split='validation',reason='No later checkpoint can strictly exceed perfect binary validation; earliest maximum wins'))
    while running(args.pid) and not (root/'selection.json').exists():
        for step in range(metadata['eval_every'],metadata['steps'],metadata['eval_every']):
            path=root/f'validation_{step:04d}.json'
            if not path.exists():continue
            try:report=read(path)
            except json.JSONDecodeError:continue
            if report['metrics']['success_rate']==report['metrics']['mean_reward']==1.:
                finalize(root,args.pid,step);return
        time.sleep(1)
    print('No early ceiling stop needed; training completed or exited',flush=True)


if __name__=='__main__':main()
