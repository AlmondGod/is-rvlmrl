"""Experimental full-state prefix serialization, outside reward/training caches.

A fresh, continuous execution supplies a trusted local Python snapshot. A second
execution restores that exact state and executes the remaining source program.
The complete recordings and non-timing outcomes must match before any reuse can
be considered. This probe never installs a training backend or awards rewards.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import random
import time

FIELDS = ('env','executor','fv','pv','evaluation','history','samples','released',
          'estimates','sheet','active','next_call','ready_at','finished','error','lastphase')


class SnapshotFailure(BaseException):
    """Infrastructure failure must bypass the simulator's measured-error catch."""


def instrument(run, checkpoint_step, snapshot, restore=False):
    """Instrument only the loop boundary, preserving the physical execution body."""
    import numpy as np
    import mujoco
    source = inspect.getsource(run)
    loop = 'for step in range(round(seconds / .02)):'
    anchor = '    started = time.monotonic()\n'
    if source.count(loop) != 1 or source.count(anchor) != 1:
        raise ValueError('Execution source changed; revalidate instrumentation')
    source = source.replace(anchor, anchor + '    _resume_step = 0\n' + (
        '    renderer.close()\n'
        '    _loaded = _load(calls, seed, profiles)\n'
        '    ' + ', '.join(FIELDS) + ' = [_loaded[k] for k in _fields]\n'
        '    _resume_step = _loaded["step"]\n'
        '    renderer = mujoco.Renderer(env.m, config["image_height"], config["image_width"])\n'
        if restore else ''))
    source = source.replace(loop, 'for step in range(_resume_step, round(seconds / .02)):\n'
        '            if step == _checkpoint and not _restore:\n'
        '                _save(locals())')
    receipt = {}
    def save(values):
        start = time.monotonic()
        bundle = {k:values[k] for k in FIELDS}
        bundle.update(step=values['step'], seed=values['seed'], profiles=values['profiles'],
                      executed_calls=values['calls'][:values['next_call']],
                      random_state=random.getstate(), numpy_state=np.random.get_state())
        temporary = snapshot.with_suffix('.partial')
        try:
            with temporary.open('xb') as f:pickle.dump(bundle,f,protocol=5)
            temporary.rename(snapshot)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise SnapshotFailure(str(error)) from error
        receipt.update(snapshot_seconds=time.monotonic()-start,snapshot_bytes=snapshot.stat().st_size,
                       checkpoint_step=values['step'],executed_calls=bundle['executed_calls'])
    def load(calls,seed,profiles):
        start=time.monotonic()
        # Only a locally generated snapshot from this probe is accepted. Pickle
        # is executable data; this API is not an untrusted archive importer.
        with snapshot.open('rb') as f:bundle=pickle.load(f)
        if (seed!=bundle['seed'] or profiles!=bundle['profiles'] or
            calls[:len(bundle['executed_calls'])]!=bundle['executed_calls']):
            raise ValueError('Restore does not have the identical physical/program prefix')
        random.setstate(bundle['random_state']);np.random.set_state(bundle['numpy_state'])
        receipt.update(restore_seconds=time.monotonic()-start,checkpoint_step=bundle['step'])
        return bundle
    namespace=dict(run.__globals__,_fields=FIELDS,_checkpoint=checkpoint_step,
                   _restore=restore,_save=save,_load=load)
    exec(compile(source,run.__code__.co_filename,'exec'),namespace)
    receipt['instrumented_source_sha256']=hashlib.sha256(source.encode()).hexdigest()
    return namespace['run'],receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--program',required=True)
    parser.add_argument('--seed',type=int,default=51001)
    parser.add_argument('--seconds',type=float,default=20.)
    parser.add_argument('--checkpoint-seconds',type=float,default=10.)
    args=parser.parse_args()
    if not 0 < args.checkpoint_seconds < args.seconds:parser.error('Checkpoint must precede end')
    from experiments.torso_assembly.acceleration import configure
    from experiments.cap_rl.assembly import run
    from analysis.compare_assembly_speed import compare
    import cv2
    cv2.setNumThreads(1);configure('vectorized')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'prefix.pickle';text=Path(args.program).read_text()
    rows=[]
    for restore in (False,True):
        fn,receipt=instrument(run,round(args.checkpoint_seconds/.02),snapshot,restore)
        start=time.monotonic()
        result=fn(args.seed,text,out/('restored' if restore else 'continuous'),seconds=args.seconds,
                  support_profile='contact_and_nails',controller_profile='southwest_approach')
        rows.append(dict(restore=restore,elapsed_seconds=time.monotonic()-start,receipt=receipt))
    comparison=compare(out/'continuous',out/'restored')
    report=dict(passes=comparison['exact'],configuration=vars(args),rows=rows,comparison=comparison,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Experimental prefix replay only; no training integration, task success, or whole-loop speed claim.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    # The large trusted scratch snapshot is not a model or reward artifact.
    # Keep the original/restored traces and instrumentation receipts instead.
    snapshot.unlink()
    if not report['passes']:raise SystemExit(1)


if __name__=='__main__':main()
