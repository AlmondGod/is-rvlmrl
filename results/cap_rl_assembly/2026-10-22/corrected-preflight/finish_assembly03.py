"""One-shot continuation of the authorized experiment; never tunes on test results."""
import datetime,json,os,pathlib,subprocess,time,traceback
root=pathlib.Path('/opt/cap-assembly');run=root/'assembly03';python=str(root/'venv/bin/python')
status=root/'assembly03-verification-status.json';tools=root/'verification-tools'
env=dict(os.environ,HF_HOME=str(root/'hf'),HF_HUB_ENABLE_HF_TRANSFER='0',MUJOCO_GL='egl',
         CAP_RL_ASSEMBLY_CACHE=str(root/'assembly03-evidence'),PYTHONPATH=str(root/'corrected-repo'))
started=time.monotonic();stages=[];stage='waiting_for_training'
def update(state,**details):
    status.write_text(json.dumps(dict(state=state,stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started,completed_stages=stages,**details),indent=2)+'\n')
def command(name,args):
    global stage
    stage=name;update('running');start=time.monotonic()
    with (root/('assembly03-'+name+'.log')).open('x') as log:
        subprocess.run([python,*args],cwd=root/'corrected-repo',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    stages.append(dict(stage=name,wall_seconds=time.monotonic()-start));update('running')
try:
    assert not status.exists()
    update('waiting')
    while not (run/'selection.json').exists():
        try:
            cmd=pathlib.Path('/proc/13729/cmdline').read_bytes()
            if b'experiments.cap_rl.train' not in cmd or b'/opt/cap-assembly/assembly03' not in cmd:
                raise RuntimeError('Training process no longer matches expected job')
        except FileNotFoundError:
            raise RuntimeError('Training exited without final selection')
        time.sleep(10)
    selection=json.loads((run/'selection.json').read_text())
    metadata=json.loads((run/'metadata.json').read_text())
    updates=[json.loads(s) for s in (run/'training.jsonl').read_text().splitlines()]
    assert len(updates)==metadata['steps']==32
    baseline=json.loads((run/'validation_baseline.json').read_text())['metrics']
    if selection['best_step']==0 or selection['validation_success_rate']<=baseline['success_rate']:
        raise RuntimeError('Validation has not established improved strict success; held-out test remains unused')
    # Wait for the trainer to exit and free GPU allocations before loading test models.
    while pathlib.Path('/proc/13729').exists():time.sleep(2)
    command('heldout-test',['-u','-m','experiments.cap_rl.train','test','--out',str(run)])
    command('record-audit',[str(tools/'cap_rl_assembly_records.py'),'--run',str(run),'--out',str(run/'record_audit.json')])
    command('generation-audit',[str(tools/'cap_rl_assembly_generation_audit.py'),'--run',str(run),'--out',str(run/'generation_audit.json')])
    command('physics-replay',['-u',str(tools/'cap_rl_assembly_test_replay.py'),'--run',str(run),'--out',str(root/'assembly03-test-replay'),'--workers','8'])
    report=json.loads((run/'test_comparison.json').read_text())
    update('complete',verified_improvement=report['delta_success_rate']>0,
           baseline=report['baseline'],trained=report['trained'],paired_wins=report['paired_wins'],paired_losses=report['paired_losses'])
except Exception as error:
    update('failed',error=type(error).__name__+':'+str(error))
    traceback.print_exc();raise
