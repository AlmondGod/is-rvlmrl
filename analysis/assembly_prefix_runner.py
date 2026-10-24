"""Experimental bounded, worker-local reuse of complete physical prefixes.

The only cache entry is a full Python/MuJoCo state, taken after the same first
nine API calls have completed. A sibling program can change its unexecuted
suffix. This is not installed into the learner. Every reuse has an explicit
sidecar receipt and requires exact seed, source, runtime and parsed-call prefix.
"""
import gc
import hashlib
import inspect
import json
from pathlib import Path
import random

import numpy as np

from analysis.assembly_memory_snapshot import clone
from analysis.benchmark_assembly_prefix import FIELDS, SnapshotFailure


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PrefixRunner:
    def __init__(self, run, provenance, prefix_calls=9):
        if type(prefix_calls) is not int or prefix_calls < 1:
            raise ValueError('A positive completed-call prefix is required')
        self.base = run
        self.source = inspect.getsource(run)
        self.provenance = provenance
        self.expected_provenance = provenance()
        self.prefix_calls = prefix_calls
        self.entry = None
        self.sources = {str(p):digest(p) for p in (
            Path(__file__),Path(__file__).with_name('assembly_memory_snapshot.py'),
            Path(__file__).with_name('benchmark_assembly_prefix.py'))}
        if self.source.count('for step in range(round(seconds / .02)):') != 1:
            raise ValueError('Physical loop changed; revalidate prefix instrumentation')

    def validate(self):
        if (inspect.getsource(self.base) != self.source or
                self.provenance() != self.expected_provenance or
                any(digest(p) != h for p,h in self.sources.items())):
            raise SnapshotFailure('Prefix implementation or physical provenance changed')

    def __call__(self, seed, text, out, **kwargs):
        from experiments.cap_rl.assembly import parse_program,profile_options
        self.validate()
        if kwargs.get('video',False):raise ValueError('Use recorded-state replay for prefix diagnostics')
        try:calls=parse_program(text)
        except (ValueError,SyntaxError):return self.base(seed,text,out,**kwargs)
        profiles=profile_options(kwargs.get('support_profile','contact_only'),
                                 kwargs.get('controller_profile','original'))
        key=dict(seed=seed,profiles=profiles,prefix=calls[:self.prefix_calls],
                 physical_provenance=self.expected_provenance)
        key=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
        seconds=kwargs.get('seconds',1200.)
        reuse=(self.entry is not None and self.entry['key']==key and
               self.entry['bundle']['step'] < round(seconds/.02))
        if not reuse:
            self.entry=None
            gc.collect()
        receipt=dict(format='experimental_full_state_prefix_v1',reused=bool(reuse),
                     seed=seed,identity_sha256=key,prefix_calls=self.prefix_calls,
                     source_sha256=self.sources,
                     base_function_sha256=hashlib.sha256(self.source.encode()).hexdigest(),
                     scope='Full physical prefix reuse, no reward prediction or policy observation from hidden state.')

        def capture(values):
            if self.entry is not None:return
            try:
                bundle={name:values[name] for name in FIELDS}
                bundle.update(step=values['step'],executed_calls=values['calls'][:values['next_call']],
                              random_state=random.getstate(),numpy_state=np.random.get_state())
                bundle,details=clone(bundle)
                self.entry=dict(key=key,bundle=bundle,origin=str(Path(out).resolve()))
                receipt.update(captured_step=bundle['step'],copy=details)
            except Exception as error:raise SnapshotFailure(str(error)) from error

        def restore():
            try:
                if not self.entry or self.entry['key']!=key:
                    raise ValueError('Prefix cache identity changed')
                bundle,details=clone(self.entry['bundle'])
                if calls[:len(bundle['executed_calls'])] != bundle['executed_calls']:
                    raise ValueError('Program prefix differs from physically executed calls')
                random.setstate(bundle['random_state']);np.random.set_state(bundle['numpy_state'])
                receipt.update(restored_step=bundle['step'],origin_episode=self.entry['origin'],copy=details)
                return bundle
            except Exception as error:raise SnapshotFailure(str(error)) from error

        anchor='    started = time.monotonic()\n'
        if self.source.count(anchor)!=1:raise SnapshotFailure('Physical setup changed')
        source=self.source.replace(anchor,anchor+'    _resume_step = 0\n'+(
            '    renderer.close()\n'
            '    _loaded = _restore_prefix()\n'
            '    '+', '.join(FIELDS)+' = [_loaded[k] for k in _fields]\n'
            '    _resume_step = _loaded["step"]\n'
            '    del _loaded\n'
            '    _collect_garbage()\n'
            '    renderer = mujoco.Renderer(env.m, config["image_height"], config["image_width"])\n'
            if reuse else ''))
        source=source.replace('for step in range(round(seconds / .02)):',
            'for step in range(_resume_step, round(seconds / .02)):\n'
            '            if active is None and next_call == _prefix_calls:\n'
            '                _capture_prefix(locals())')
        namespace=dict(self.base.__globals__,_fields=FIELDS,_restore_prefix=restore,
                       _capture_prefix=capture,_prefix_calls=self.prefix_calls,
                       _collect_garbage=gc.collect)
        exec(compile(source,self.base.__code__.co_filename,'exec'),namespace)
        receipt['instrumented_source_sha256']=hashlib.sha256(source.encode()).hexdigest()
        try:
            result=namespace['run'](seed,text,out,**kwargs)
            self.validate()
        except BaseException:
            self.entry=None
            raise
        (Path(out)/'prefix_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return result
