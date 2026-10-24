"""Submit independent reward groups without a coordinator-wide batch barrier.

This module only schedules unchanged physical evaluations. The caller owns
policy versions, old log-probabilities, advantages, staleness limits and update
ordering. No training algorithm is changed by importing this module.
"""
from concurrent.futures import Future, ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path


@dataclass
class PendingBatch:
    queue: object
    entries: list

    def ready(self):
        return all(future.done() for future, _, _ in self.entries)

    def result(self):
        self.queue.validate()
        return [dict(future.result(), batch_duplicate=duplicate, inflight_shared=shared)
                for future, duplicate, shared in self.entries]

    def futures(self):
        return {future for future, _, _ in self.entries}


class RewardQueue:
    """One owner thread; submit all groups before waiting on any group's reward."""
    def __init__(self, executor, execute, identity, shortcut, validate=lambda: None):
        self.executor, self.execute = executor, execute
        self.identity, self.shortcut, self.validate = identity, shortcut, validate
        self.inflight = {}

    def submit(self, layouts, texts):
        if len(layouts) != len(texts):
            raise ValueError('Each program needs one layout')
        self.validate()
        entries, seen = [], set()
        for layout, text in zip(layouts, texts):
            key = self.identity(layout, text)
            duplicate = key in seen
            seen.add(key)
            future = self.inflight.get(key)
            shared = future is not None and not future.done()
            if not shared:
                # Finished evaluations go back through the checked physical
                # cache. Never reuse a stale in-memory result across submissions.
                immediate = self.shortcut(layout, text, key)
                if immediate is None:
                    future = self.executor.submit(self.execute, layout, text)
                else:
                    future = Future(); future.set_result(immediate)
                self.inflight[key] = future
            entries.append((future, duplicate, shared))
        self.inflight = {k:f for k,f in self.inflight.items() if not f.done()}
        return PendingBatch(self, entries)

    @staticmethod
    def wait_ready(batches, timeout=30.):
        """Bounded event wait; the caller decides which ready group to consume."""
        ready = [b for b in batches if b.ready()]
        if ready: return ready
        pending = {f for b in batches for f in b.futures() if not f.done()}
        if pending: wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
        return [b for b in batches if b.ready()]


def check_memory_budget(workers, cgroup=Path('/sys/fs/cgroup')):
    """Conservative admission estimate, not a replacement for one pool owner.

    The measured assembly workers use up to about 1.9 GB RSS. Reserve 2.25 GB
    per new worker plus 3 GB for temporary buffers. Inactive file pages are
    reclaimable; existing anonymous allocations must remain in the budget.
    """
    maximum = Path(cgroup)/'memory.max'
    if not maximum.exists() or maximum.read_text().strip() == 'max':
        return dict(bounded=False, reason='No finite cgroup-v2 memory limit available')
    limit = int(maximum.read_text())
    current = int((Path(cgroup)/'memory.current').read_text())
    stats = dict(line.split() for line in (Path(cgroup)/'memory.stat').read_text().splitlines())
    working = max(0, current-int(stats.get('inactive_file', 0)))
    required = workers*2_250_000_000+3_000_000_000
    if limit-working < required:
        raise RuntimeError(f'Insufficient host RAM for {workers} new assembly workers: '
                           f'{limit-working} bytes available, {required} bytes reserved')
    return dict(bounded=True, limit_bytes=limit, working_set_bytes=working,
                reserved_bytes=required, workers=workers)


def assembly_queue(task, workers=8, threads=1):
    """Use the existing frozen worker, provenance and certified reward cache."""
    manifest = task.provenance()
    if task.EXPECTED_PROVENANCE is None or task.EXPECTED_PROVENANCE != manifest:
        raise ValueError('Freeze physical provenance before scheduling rewards')
    if workers < 1 or threads < 1: raise ValueError('Positive worker sizes required')
    if task._POOL is not None:
        raise RuntimeError('Shut down the existing assembly pool before creating another')
    budget = check_memory_budget(workers)
    root = Path(os.environ.get('CAP_RL_ASSEMBLY_CACHE', 'artifacts/assembly_rollouts')).resolve()
    def validate():
        if task.provenance() != manifest or task.EXPECTED_PROVENANCE != manifest:
            raise RuntimeError('Physical source changed while rewards were pending')
    def identity(layout, text):
        try: calls = task.parse_program(text)
        except (ValueError, SyntaxError):
            return ('invalid', layout.seed, text)
        value = dict(seed=layout.seed, calls=calls, provenance=manifest)
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    def shortcut(layout, text, key):
        if isinstance(key, tuple) or (root/key/'certificate.json').exists():
            return task.execute(layout, text)
        return None
    executor = ProcessPoolExecutor(workers,
        mp_context=multiprocessing.get_context('spawn'), initializer=task._worker_init,
        initargs=(manifest, task.SUPPORT_PROFILE, task.CONTROLLER_PROFILE, task.ASSEMBLY_BACKEND, threads))
    queue = RewardQueue(executor, task.execute, identity, shortcut, validate)
    queue.memory_budget = budget
    return queue
