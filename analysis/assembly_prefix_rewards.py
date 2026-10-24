"""Versioned, certified rewards with worker-local complete-prefix reuse.

Candidates with identical parsed physical prefixes run consecutively in one
worker. The existing evaluator, certificate format, cache locking and strict
reward semantics remain in use. This optional adapter must be installed in a
fresh process and included in every training/evaluation provenance receipt.
"""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

_TASK = None
_ORIGINAL_EXECUTE = None
_PROFILE = None


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def profile(prefix_calls=9):
    if prefix_calls!=9:raise ValueError('This adapter reuses only the nine completed member-placement calls')
    names=('assembly_prefix_rewards.py','assembly_prefix_runner.py','assembly_prefix_evidence.py',
           'assembly_memory_snapshot.py','benchmark_assembly_prefix.py')
    return dict(format='complete_native_prefix_rewards_v1',prefix_calls=prefix_calls,
                sources_sha256={name:sha(Path(__file__).with_name(name)) for name in names},
                cache_entries_per_worker=1,origin_evidence_required=True)


def verify_episode_prefix(directory,expected_profile):
    """Independently bind reused recorded prefixes to their complete origin."""
    import numpy as np
    from experiments.cap_rl.assembly import parse_program
    from analysis.assembly_prefix_evidence import verify_origin
    from analysis.benchmark_assembly_prefix import SnapshotFailure
    root=Path(directory);receipt_path=root/'prefix_receipt.json'
    if not receipt_path.exists():
        raise SnapshotFailure('Physical prefix-profile episode lacks execution receipt')
    receipt=json.loads(receipt_path.read_text())
    if receipt['prefix_calls']!=expected_profile['prefix_calls']:
        raise SnapshotFailure('Recorded prefix boundary changed')
    expected=expected_profile['sources_sha256']
    required=set(expected)-{'assembly_prefix_rewards.py'}
    if {Path(path).name for path in receipt['source_sha256']}!=required:
        raise SnapshotFailure('Prefix snapshot source receipt is incomplete')
    for path,digest in receipt['source_sha256'].items():
        if Path(path).name not in expected or expected[Path(path).name]!=digest:
            raise SnapshotFailure('Prefix snapshot source differs from reward provenance')
    if not receipt['reused']:return receipt
    origin=Path(receipt['origin_episode'])
    verify_origin(dict(origin=str(origin),origin_files_sha256=receipt['origin_files_sha256']))
    old=json.loads((origin/'prefix_receipt.json').read_text())
    if old['reused'] or old.get('captured_step')!=receipt['restored_step']:
        raise SnapshotFailure('Prefix origin is not the recorded cold capture')
    before=json.loads((origin/'result.json').read_text());after=json.loads((root/'result.json').read_text())
    count=expected_profile['prefix_calls']
    if (before['seed']!=after['seed'] or
        json.loads((origin/'provenance.json').read_text())!=json.loads((root/'provenance.json').read_text()) or
        parse_program((origin/'program.py').read_text())[:count]!=parse_program((root/'program.py').read_text())[:count]):
        raise SnapshotFailure('Prefix seed, physics or parsed calls differ')
    samples=(receipt['restored_step']+4)//5
    with np.load(origin/'states.npz') as a,np.load(root/'states.npz') as b:
        if (min(len(a['rows']),len(b['rows']))<samples or
                not np.array_equal(a['rows'][:samples],b['rows'][:samples])):
            raise SnapshotFailure('Recorded physical prefix differs from its origin')
    return receipt


def execute(layout,text):
    if _TASK is None:raise RuntimeError('Prefix reward worker is not initialized')
    if profile()!=_PROFILE:raise RuntimeError('Prefix reward implementation changed')
    result=_ORIGINAL_EXECUTE(layout,text)
    if result.get('evidence') and result.get('physics_started'):
        verify_episode_prefix(result['evidence'],_PROFILE)
    return result


def install(task):
    global _TASK,_ORIGINAL_EXECUTE,_PROFILE
    from experiments.cap_rl import assembly
    from analysis.assembly_prefix_evidence import EvidenceBoundPrefixRunner
    current=profile()
    if _TASK is not None:
        if _TASK is not task or _PROFILE!=current or task.execute is not execute:
            raise RuntimeError('Prefix reward adapter changed or installed twice differently')
        return current
    if task._POOL is not None:raise RuntimeError('Install prefix rewards before starting physics workers')
    original=assembly.provenance
    def provenance(**kwargs):
        if profile()!=current:raise RuntimeError('Prefix reward sources changed')
        return dict(original(**kwargs),execution_prefix=current)
    assembly.provenance=provenance
    task._provenance=provenance
    runner=EvidenceBoundPrefixRunner(assembly.run,task.provenance,prefix_calls=9)
    _TASK=task;_ORIGINAL_EXECUTE=task.execute;_PROFILE=current
    task.run=runner;task.execute=execute;task._worker_init=worker_init
    return current


def worker_init(expected,support,controller,backend,threads):
    import cv2
    from experiments.cap_rl import task_assembly as task
    cv2.setNumThreads(threads);task.configure(support,controller,backend)
    install(task);task.EXPECTED_PROVENANCE=expected
    if task.provenance()!=expected:raise RuntimeError('Worker prefix/physical provenance differs')


def execute_group(layouts,texts):
    if len(layouts)!=len(texts):raise ValueError('Incomplete physical program group')
    return [execute(layout,text) for layout,text in zip(layouts,texts)]


def group_programs(layouts,texts,prefix_calls=9):
    from experiments.cap_rl.assembly import parse_program
    if len(layouts)!=len(texts):raise ValueError('Each program needs one layout')
    groups=OrderedDict()
    for index,(layout,text) in enumerate(zip(layouts,texts)):
        try:key=(layout.seed,tuple(parse_program(text)[:prefix_calls]))
        except (ValueError,SyntaxError):key=('invalid',index)
        groups.setdefault(key,[]).append((index,layout,text))
    return list(groups.values())


class PendingGroups:
    def __init__(self,groups,size,validate):
        self.groups=groups;self.size=size;self.validate=validate
    def ready(self):return all(f.done() for _,f in self.groups)
    def futures(self):return {f for _,f in self.groups}
    def result(self):
        from experiments.cap_rl.assembly import parse_program
        self.validate();rows=[None]*self.size
        for group,future in self.groups:
            values=future.result()
            if len(values)!=len(group):raise RuntimeError('Incomplete physical reward group')
            seen=set()
            for (index,layout,text),value in zip(group,values):
                try:key=(layout.seed,tuple(parse_program(text)))
                except (ValueError,SyntaxError):key=(layout.seed,text)
                rows[index]=dict(value,batch_duplicate=key in seen,inflight_shared=False)
                seen.add(key)
        if any(row is None for row in rows):raise RuntimeError('Missing physical reward')
        return rows


def memory_budget(workers):
    root=Path('/sys/fs/cgroup')
    if not (root/'memory.max').exists() or (root/'memory.max').read_text().strip()=='max':
        raise RuntimeError('Prefix worker deployment requires an explicit cgroup memory budget')
    limit=int((root/'memory.max').read_text())
    stats=dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
    working=sum(int(stats.get(k,0)) for k in ('anon','kernel','shmem'))
    required=workers*3_500_000_000+3_000_000_000
    if limit-working<required:
        raise RuntimeError(f'Insufficient host memory: {limit-working} available, {required} reserved')
    return dict(limit_bytes=limit,working_bytes=working,reserved_bytes=required,workers=workers,
                accounting='Anonymous/kernel/shared memory plus 3.5 GB per new prefix worker and 3 GB transient reserve; clean file cache is reclaimable.')


class GroupedPrefixQueue:
    def __init__(self,task,workers=6,threads=1):
        if workers<1 or threads<1:raise ValueError('Positive worker/thread counts required')
        install(task)
        self.task=task;self.manifest=task.provenance()
        if task.EXPECTED_PROVENANCE!=self.manifest or task._POOL is not None:
            raise RuntimeError('Freeze prefix provenance and use only one pool owner')
        self.memory_budget=memory_budget(workers)
        self.executor=ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=worker_init,initargs=(self.manifest,task.SUPPORT_PROFILE,
                task.CONTROLLER_PROFILE,task.ASSEMBLY_BACKEND,threads))
    def validate(self):
        if self.task.provenance()!=self.manifest or self.task.EXPECTED_PROVENANCE!=self.manifest:
            raise RuntimeError('Physical/prefix sources changed with queued rewards')
    def submit(self,layouts,texts):
        self.validate();pending=[]
        for group in group_programs(layouts,texts):
            future=self.executor.submit(execute_group,[r[1] for r in group],[r[2] for r in group])
            pending.append((group,future))
        return PendingGroups(pending,len(layouts),self.validate)
    def close(self):self.executor.shutdown(wait=True)
