"""Training/validation-only futility decision; never reads held-out outcomes."""
import datetime,json,os,pathlib,signal,time
root=pathlib.Path('/opt/cap-assembly/assembly03');pid,group=13729,13725
policy=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    criterion='At update-8 validation, strict success zero AND updates 5..8 have zero strict training success and zero within-group reward variance.',
    action='Retire dense optimizer before any held-out test; preserve all snapshots and restart raw with success-only reward. Physics, controller, prompts, splits and gates stay fixed.',
    next_candidate=dict(optimization_reward='strict_success',lr=3e-5,steps=16,eval_every=2,raw_initialization=True))
policy_path=root/'futility_policy.json';assert not policy_path.exists();assert not (root/'retired.json').exists()
assert not (root/'validation_0008.json').exists(), 'Declare the rule before this validation result'
policy_path.write_text(json.dumps(policy,indent=2)+'\n')
while True:
    command=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    assert b'experiments.cap_rl.train' in command and str(root).encode() in command and os.getpgid(pid)==group
    try:validation=json.loads((root/'validation_0008.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(.5);continue
    updates=[json.loads(s) for s in (root/'training.jsonl').read_text().splitlines()]
    recent=[r for r in updates if 5<=r['step']<=8]
    stalled=(validation['metrics']['successes']==0 and len(validation['rows'])==4 and len(recent)==4 and
             all(r['success_rate']==0 and r['active_groups']==0 for r in recent))
    if not stalled:
        print('Futility criterion not met; keep original 32-update run.',flush=True);break
    os.killpg(group,signal.SIGSTOP)
    try:
        updates=[json.loads(s) for s in (root/'training.jsonl').read_text().splitlines()]
        if updates[-1]['step']!=8:
            os.killpg(group,signal.SIGCONT)
            print('Missed exact boundary; keeping original run.',flush=True);break
        assert (root/'checkpoints/step-0008/checkpoint.json').exists()
        assert not (root/'test_comparison.json').exists()
        rows=[json.loads(s) for s in (root/'rollouts.jsonl').read_text().splitlines()]
        assert len(rows)==64
        elapsed=float(pathlib.Path('/proc/uptime').read_text().split()[0])-int(pathlib.Path(f'/proc/{pid}/stat').read_text().split()[21])/os.sysconf('SC_CLK_TCK')
        record=dict(status='retired_dense_reward_pilot',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            reason='Confirmed uniform partial-program collapse and no strict validation improvement. Dense group-relative credit can reinforce incomplete programs; next candidate uses strict-success-only learning reward.',
            completed_updates=8,planned_updates=32,last_metric=updates[-1],validation=validation['metrics'],
            trainer_elapsed_seconds=elapsed,trained_weights_retained=True,retained_checkpoint='checkpoints/step-0008',
            selection_completed=False,test_evaluated=False,futility_policy=policy,
            timing_scope='Last metric excludes the final validation; process elapsed includes setup, baseline and final validation.')
        (root/'retired.json').write_text(json.dumps(record,indent=2)+'\n')
        os.killpg(group,signal.SIGTERM);os.killpg(group,signal.SIGCONT)
        print(json.dumps(record),flush=True)
    except BaseException:
        os.killpg(group,signal.SIGCONT);raise
    break
