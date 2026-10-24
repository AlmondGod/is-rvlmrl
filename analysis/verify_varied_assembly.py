"""Run the predeclared frozen-policy verification after perfect validation.

No training, decoder selection, reward changes or test-guided retries. A stage
failure stops the sequence and preserves its log; successful stages are bound
to receipts so a resumed invocation cannot silently skip changed results.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from analysis.finalize_assembly_ceiling import running, save, sha


def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())


def eligible(root):
    metadata, selection = read(root/'metadata.json'), read(root/'selection.json')
    if (metadata['assembly_split'] != 'varied_v2' or
        metadata.get('policy_decoding') != 'assembly_permutations_v1' or
        selection['selection_split'] != 'validation' or
        selection['validation_success_rate'] != 1. or selection['best_step'] <= 0):
        raise ValueError('Automatic final evaluation requires a learned policy selected at 8/8 varied validation')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--replay', required=True)
    p.add_argument('--wait-for-pid', type=int)
    a=p.parse_args();root=Path(a.run).resolve();replay=Path(a.replay).resolve()
    source=root/'verification_source'/Path(__file__).name;source.parent.mkdir(exist_ok=True)
    if source.exists() and sha(source) != sha(__file__):
        raise ValueError('Verification source changed; preserve and explicitly review the earlier supervisor')
    source.write_bytes(Path(__file__).read_bytes())
    status_path=root/'verification_status.json'
    old=read(status_path) if status_path.exists() else {}
    completed=old.get('completed_stages', [])
    status=dict(started_utc=old.get('started_utc',utc()), completed_stages=completed,
                source_sha256=sha(source), replay_path=str(replay), state='waiting_for_selection', utc=utc())
    if old and (old.get('source_sha256') != status['source_sha256'] or old.get('replay_path') != str(replay)):
        raise ValueError('Supervisor invocation differs from its saved receipt')
    save(status_path,status)
    if a.wait_for_pid:
        while running(a.wait_for_pid): time.sleep(10)
        # The ceiling finalizer writes selection only after stopping training.
        for _ in range(30):
            if (root/'selection.json').exists():break
            time.sleep(1)
    try:
        metadata=eligible(root)
        frozen_names=('metadata.json','selection.json','initial_adapter/adapter_model.safetensors',
                      'best_adapter/adapter_model.safetensors')
        frozen={name:sha(root/name) for name in frozen_names}
        if old.get('selected_inputs_sha256',frozen) != frozen:
            raise ValueError('Selected policy changed after verification began')
        status['selected_inputs_sha256']=frozen
        logs=root/'verification_logs';logs.mkdir(exist_ok=True)
        stages=[
            ('pretest', 'analysis.cap_rl_assembly_records',
             ['--run',str(root),'--out',str(root/'pretest_record_audit.json'),'--pretest'], root/'pretest_record_audit.json'),
            ('primary_test', 'analysis.cap_rl_assembly_grammar',
             ['test','--out',str(root)], root/'test_comparison.json'),
            ('raw_baseline', 'analysis.cap_rl_assembly_raw_baseline',
             ['--run',str(root)], root/'raw_test_baseline.json'),
            ('unconstrained_transfer', 'analysis.cap_rl_assembly_unconstrained',
             ['--run',str(root),'--split','test'], root/'unconstrained_test_comparison.json'),
            ('record_audit', 'analysis.cap_rl_assembly_records',
             ['--run',str(root),'--out',str(root/'record_audit.json')], root/'record_audit.json'),
            ('generation_audit', 'analysis.cap_rl_assembly_generation_audit',
             ['--run',str(root),'--out',str(root/'generation_audit.json')], root/'generation_audit.json'),
            ('physical_replay', 'analysis.cap_rl_assembly_test_replay',
             ['--run',str(root),'--out',str(replay),'--seeds','53001','53012','53024',
              '--workers',str(metadata['physics_workers']),'--physics-threads',str(metadata['physics_threads'])], replay/'audit.json')]
        from analysis.assembly_visual_prompt import enabled, verify_sources
        if enabled(metadata):
            verify_sources(root,metadata)
            visual_stages={'primary_test','raw_baseline','unconstrained_transfer','generation_audit'}
            stages=[(name,'analysis.assembly_visual_stage',['--run',str(root),'--stage',name],output)
                    if name in visual_stages else (name,module,arguments,output)
                    for name,module,arguments,output in stages]
            stages.insert(-1,('observation_audit','analysis.assembly_visual_observation_audit',
                ['--run',str(root),'--out',str(root/'observation_audit.json')],root/'observation_audit.json'))
        for index,(name,module,arguments,output) in enumerate(stages):
            if {name:sha(root/name) for name in frozen_names} != frozen:
                raise ValueError('Frozen selection changed during verification')
            if index < len(completed):
                receipt=completed[index]
                if receipt['name'] != name or receipt['output_sha256'] != sha(output):
                    raise ValueError('Completed verification stage changed: '+name)
                continue
            if output.exists():
                raise FileExistsError('Unreceipted output must be reviewed, not silently reused: '+str(output))
            status.update(state='running',stage=name,utc=utc());save(status_path,status)
            print(json.dumps(dict(stage=name,state='started',utc=utc())),flush=True)
            started=time.monotonic();log=logs/(name+'.log')
            with log.open('w') as stream:
                subprocess.run([sys.executable,'-u','-m',module,*arguments],
                               stdout=stream,stderr=subprocess.STDOUT,check=True)
            report=read(output)
            if name.endswith('audit') or name in ('pretest','physical_replay'):
                if report.get('passes') is not True: raise ValueError('Verification failed: '+name)
            completed.append(dict(name=name,output=str(output),output_sha256=sha(output),
                                  log_sha256=sha(log),seconds=time.monotonic()-started,utc=utc()))
            save(status_path,status)
            print(json.dumps(dict(stage=name,state='complete',utc=utc())),flush=True)
        status.update(state='complete',stage=None,utc=utc());save(status_path,status)
    except Exception as error:
        status.update(state='failed',error=repr(error),utc=utc());save(status_path,status)
        raise


if __name__=='__main__': main()
