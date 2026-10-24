import datetime,json,os,pathlib,signal,time
root=pathlib.Path('/opt/cap-assembly/assembly02')
pid,group=5029,4836
expected='experiments.cap_rl.train'
record=root/'retired.json'
assert not record.exists()
while True:
    cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert expected in cmd and '/opt/cap-assembly/assembly02' in cmd and os.getpgid(pid)==group
    metrics=[json.loads(line) for line in (root/'training.jsonl').read_text().splitlines() if line.strip()]
    rows=[json.loads(line) for line in (root/'rollouts.jsonl').read_text().splitlines() if line.strip()]
    last=metrics[-1]['step']
    if last>=4 and sum(r['step']==last for r in rows)==8:
        # Stop only this verified job group. SSH, other jobs and the pod survive.
        os.killpg(group,signal.SIGSTOP)
        result=dict(status='retired_exploratory_pilot',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            reason='Confirmed contact-only support reward omits actual internal nail forces. Restart raw Qwen under a separately frozen corrected profile; do not mix reward definitions.',
            completed_updates=last,last_metric=metrics[-1],trainer_cmdline=cmd,process_group=group,
            trained_weights_retained=False,selection_completed=False,test_evaluated=False,
            limitation='No trained checkpoint was scheduled before update 8. This pilot is discarded as an initialization; logs and physical evidence are preserved.',
            trainer_elapsed_seconds=float(pathlib.Path('/proc/uptime').read_text().split()[0])-int(pathlib.Path(f'/proc/{pid}/stat').read_text().split()[21])/os.sysconf('SC_CLK_TCK'))
        record.write_text(json.dumps(result,indent=2)+'\n')
        os.killpg(group,signal.SIGTERM)
        os.killpg(group,signal.SIGCONT)
        print(json.dumps(result),flush=True)
        break
    time.sleep(1)
