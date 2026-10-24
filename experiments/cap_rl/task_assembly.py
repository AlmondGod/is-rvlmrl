"""Offline full-program CaP-RL adapter for the three-arm physical benchmark."""
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path

from .assembly import parse_program, provenance as _provenance, profile_options, run

SYSTEM = 'Write only the requested Python API program. Do not explain it.'
EXPECTED_PROVENANCE = None
_FAILURES = {}
PHYSICS_WORKERS = 1
PHYSICS_THREADS = 1
DATASET_PROFILE = 'pilot_v1'
_POOL = None
_BATCH_EXECUTOR = None
SUPPORT_PROFILE = 'contact_only'
CONTROLLER_PROFILE = 'original'
ASSEMBLY_BACKEND = 'original'


def configure(support_profile='contact_only', controller_profile='original', assembly_backend='original'):
    global SUPPORT_PROFILE, CONTROLLER_PROFILE, ASSEMBLY_BACKEND
    profile_options(support_profile, controller_profile)
    if _POOL is not None and (support_profile, controller_profile, assembly_backend) != (SUPPORT_PROFILE, CONTROLLER_PROFILE, ASSEMBLY_BACKEND):
        raise RuntimeError('Cannot change assembly profiles with active physics workers')
    from experiments.torso_assembly.acceleration import configure as configure_backend
    configure_backend(assembly_backend)
    SUPPORT_PROFILE, CONTROLLER_PROFILE = support_profile, controller_profile
    ASSEMBLY_BACKEND = assembly_backend


def provenance():
    return _provenance(support_profile=SUPPORT_PROFILE, controller_profile=CONTROLLER_PROFILE)


def terminal_failure_prefix(result, calls):
    """Only a measured terminal runtime failure makes later code irrelevant."""
    history = result.get('history', [])
    if (not str(result.get('error', '')).startswith(('RuntimeError:', 'ValueError:')) or
            not history or history[-1]['status'] != 'failed'):
        return None
    prefix = [(h['skill'], h['index']) for h in history]
    if prefix != calls[:len(prefix)]:
        raise RuntimeError('Measured history differs from parsed program')
    return prefix


@dataclass(frozen=True)
class Layout:
    seed: int

    @property
    def key(self):
        return f'assembly-full-v1-{self.seed}'


def datasets():
    # Fresh layouts, distinct from historical development and model pilots.
    # The policy does not receive the seed. The same program must work through
    # supplied RGB feedback skills under lane, XY and yaw randomization.
    if DATASET_PROFILE == 'varied_v2':
        return {'train': [Layout(i) for i in range(51001, 51033)],
                'validation': [Layout(i) for i in range(52001, 52009)],
                'test': [Layout(i) for i in range(53001, 53025)]}
    if DATASET_PROFILE != 'pilot_v1':
        raise ValueError('Unknown assembly dataset profile')
    return {'train': [Layout(i) for i in range(41001, 41009)],
            'validation': [Layout(i) for i in range(42001, 42005)],
            'test': [Layout(i) for i in range(43001, 43009)]}


def prompt(layout):
    return '''Write a complete program for three robot arms to assemble a wooden frame with plywood and flip it.
The four wooden members and plywood start loose. Two arms have grippers and the third has a nail tool. Each API call runs a supplied camera-guided feedback skill to completion. Grasping uses physical contacts; connections are finite-strength simulated nails. Stop automatically on any failed skill. Six seconds of physical settling occur before each call.
Member identities: 0 long left-side member -> front; 1 short left-side member -> left; 2 long right-side member -> rear; 3 short right-side member -> right.
API: only these calls, with literal integer indices 0..3 where specified:
position_member(i): pick, lift, carry and align member i; keeps holding it.
release_member(i): release the positioned member and withdraw; one release per member.
park_grippers(): after all four members are released, park both arms clear of the nail tool.
nail_corner(i): visit corner i once (0 southwest, 1 southeast, 2 northeast, 3 northwest); requires all four members released and grippers parked.
feed_panel(): after four corner visits, feed the plywood sheet into the workspace.
position_panel(): after feeding, grasp the sheet with both grippers, lift it off its supports, retract shelves, align and lower; keeps holding it.
release_panel(): release the positioned sheet and withdraw.
nail_panel(i): visit the sheet-to-member i connection once; requires sheet released.
flip_structure(): after all eight nail visits, grasp, lift clear, turn 180 degrees and hold inverted.
finish(): declare completion; must be the last call.
stop(): end an unsuccessful attempt.
All four members must form a released frame before corner nailing. The sheet must be placed and released before sheet nailing. Success requires all eight connections intact, correct frame/sheet geometry, release and stability before lifting, and the entire structure retained during a five-second inverted hold. Triggering nails alone does not ensure success.
Output only API calls, at most 32. No loops, variables, imports, functions, calculations, prints, or explanations.'''


@contextmanager
def execution_lock(root, key):
    """One physical execution per identity, even across independent processes."""
    locks = root / '.locks'
    locks.mkdir(parents=True, exist_ok=True)
    # Keep lock files: deleting one can let a newcomer lock a different inode.
    with (locks / (key + '.lock')).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def execute(layout, text):
    try:
        calls = parse_program(text)
    except (SyntaxError, ValueError) as exc:
        return dict(reward=0., success=False, valid=False, error=str(exc), physics_started=False)
    # No predicted rewards, reset-state injection, or version transfer. A
    # previously measured terminal failure can also be reused when all calls
    # through that failure are identical: no later instruction can execute.
    manifest = provenance()
    if EXPECTED_PROVENANCE is not None and manifest != EXPECTED_PROVENANCE:
        raise RuntimeError('Assembly dependencies changed during this training/evaluation run')
    identity = dict(seed=layout.seed, calls=calls, provenance=manifest)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(os.environ.get('CAP_RL_ASSEMBLY_CACHE', 'artifacts/assembly_rollouts')).resolve()
    context = hashlib.sha256(json.dumps(dict(seed=layout.seed, provenance=manifest,
                                           cache_root=str(root)), sort_keys=True).encode()).hexdigest()
    requested_key = key
    for prefix, measured_key in _FAILURES.get(context, []):
        if calls[:len(prefix)] == prefix:
            key = measured_key
            break
    out = root / key
    certificate = out / 'certificate.json'
    with execution_lock(root, key):
        if provenance() != manifest:
            raise RuntimeError('Assembly dependencies changed while waiting for physics')
        reused = certificate.exists()
        if reused:
            cert = json.loads(certificate.read_text())
            for name, sha in cert['files'].items():
                if hashlib.sha256((out / name).read_bytes()).hexdigest() != sha:
                    raise RuntimeError('Cached assembly evidence changed: ' + name)
            if cert['identity_sha256'] != key:
                raise RuntimeError('Cached assembly identity mismatch')
            result = json.loads((out / 'result.json').read_text())
            if key != requested_key and terminal_failure_prefix(result, calls) is None:
                raise RuntimeError('Failure-prefix cache lacks a measured terminal failure')
        else:
            # Existing incomplete runs are errors, never silently retried/relabelled.
            result = run(layout.seed, text, out,
                         support_profile=SUPPORT_PROFILE, controller_profile=CONTROLLER_PROFILE)
            if result.get('error') == 'interrupted':
                raise RuntimeError('Interrupted physics is not a measured training reward')
            if provenance() != manifest:
                raise RuntimeError('Assembly dependencies changed during physical execution')
            cert = dict(identity_sha256=key, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in out.iterdir() if p.is_file()})
            certificate.write_text(json.dumps(cert, indent=2))
    prefix = terminal_failure_prefix(result, calls)
    if prefix is not None and not any(p == prefix for p, _ in _FAILURES.get(context, [])):
        _FAILURES.setdefault(context, []).append((prefix, key))
    summary = {k: v for k, v in result.items() if k not in ('history', 'final_state', 'last_flip')}
    summary.update(evidence=str(out), evidence_id=key, cache_hit=reused,
                   cache_reuse='terminal_failure_prefix' if key != requested_key else 'exact_program',
                   result_sha256=cert['files']['result.json'])
    return summary


def _worker_init(expected, support_profile, controller_profile, assembly_backend='original', threads=1):
    global EXPECTED_PROVENANCE
    import cv2
    cv2.setNumThreads(threads)
    configure(support_profile, controller_profile, assembly_backend)
    EXPECTED_PROVENANCE = expected


def execute_batch(layouts, texts):
    """Parallel independent physical executions, with exact AST deduplication."""
    global _POOL
    if len(layouts) != len(texts):
        raise ValueError('Each generated program needs exactly one layout')
    if PHYSICS_WORKERS == 1:
        return [execute(layout, text) for layout, text in zip(layouts, texts)]
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    if _POOL is None:
        _POOL = ProcessPoolExecutor(PHYSICS_WORKERS,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_worker_init, initargs=(EXPECTED_PROVENANCE, SUPPORT_PROFILE, CONTROLLER_PROFILE, ASSEMBLY_BACKEND, PHYSICS_THREADS))
    manifest = provenance()
    if EXPECTED_PROVENANCE is not None and manifest != EXPECTED_PROVENANCE:
        raise RuntimeError('Assembly dependencies changed during this training/evaluation run')
    root = Path(os.environ.get('CAP_RL_ASSEMBLY_CACHE', 'artifacts/assembly_rollouts')).resolve()
    groups, results = {}, [None] * len(texts)
    for i, (layout, text) in enumerate(zip(layouts, texts)):
        try:
            calls = parse_program(text)
        except (SyntaxError, ValueError):
            results[i] = execute(layout, text)
            continue
        identity = json.dumps(dict(seed=layout.seed, calls=calls, provenance=manifest), sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()
        context = hashlib.sha256(json.dumps(dict(seed=layout.seed, provenance=manifest,
                                      cache_root=str(root)), sort_keys=True).encode()).hexdigest()
        if (root / key / 'certificate.json').exists() or any(
                calls[:len(prefix)] == prefix for prefix, _ in _FAILURES.get(context, [])):
            results[i] = execute(layout, text)
        else:
            groups.setdefault(key, []).append((i, layout, text, calls, context))
    futures = [(group, _POOL.submit(execute, group[0][1], group[0][2])) for group in groups.values()]
    for group, future in futures:
        result = future.result()  # Infrastructure errors fail the run, never become reward zero.
        measured = json.loads((Path(result['evidence']) / 'result.json').read_text())
        for n, (i, layout, text, calls, context) in enumerate(group):
            results[i] = dict(result, batch_duplicate=n > 0)
            prefix = terminal_failure_prefix(measured, calls)
            if prefix is not None and not any(p == prefix for p, _ in _FAILURES.get(context, [])):
                _FAILURES.setdefault(context, []).append((prefix, result['evidence_id']))
    return results


def submit_batch(layouts, texts):
    """Start physics while the unchanged GPU policy computes rollout log-probs.

    A single coordinator owns the persistent process pool and failure cache.
    No update occurs until the caller collects every physical reward.
    """
    global _BATCH_EXECUTOR
    from concurrent.futures import ThreadPoolExecutor
    if _BATCH_EXECUTOR is None:
        _BATCH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='assembly-rollouts')
    return _BATCH_EXECUTOR.submit(execute_batch, list(layouts), list(texts))


def shutdown_workers():
    global _POOL, _BATCH_EXECUTOR
    if _BATCH_EXECUTOR is not None:
        _BATCH_EXECUTOR.shutdown(wait=True)
        _BATCH_EXECUTOR = None
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL = None
