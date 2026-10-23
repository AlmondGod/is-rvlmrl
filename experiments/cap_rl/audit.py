"""Replay saved test programs and independently verify the reported improvement.

Symbolic tasks need only the standard library. Physics tasks also need MuJoCo
and NumPy, but no GPU or model weights.
It verifies execution and accounting, not whether the claimed model emitted text.
Model-generation provenance is provided by the checkpoint and saved source/config.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import json
from pathlib import Path
import random

from .tasks import get_task


def equivalent(a, b, tolerance=0.):
    """Preserve exact symbolic checks, allow tiny cross-platform physics roundoff."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k], tolerance) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(equivalent(x, y, tolerance) for x, y in zip(a, b))
    if type(a) in (float, int) and type(b) in (float, int):
        return abs(a-b) <= tolerance
    return a == b


def init_test_worker(task_name):
    global worker_task, worker_layouts
    worker_task = get_task(task_name)
    worker_layouts = {x.key: x for x in worker_task.datasets()['test']}


def test_replay_worker(item):
    label, index, layout_id, completion, truncated = item
    replay = worker_task.execute(worker_layouts[layout_id], completion)
    if truncated:
        replay.update(reward=0., success=False, valid=False, error='truncated')
    return (label, index), json.loads(json.dumps(replay))


def audit(directory, workers=1):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    report = json.loads((directory / "test_comparison.json").read_text())
    task = get_task(metadata.get("task", "pick_place"))
    physics = metadata.get("task") in ("mujoco_docking", "robot_lift", "frame_place", "frame_place_ordered")
    if physics:
        import mujoco
        if mujoco.__version__ != metadata["physics_runtime"]["mujoco"]:
            raise ValueError("MuJoCo runtime version differs from training")
    task_path = Path(task.__file__)
    dependencies = [task_path]
    if metadata.get("task") == "ordered_two":
        dependencies.append(task_path.with_name("task.py"))  # shared safe grammar
    if metadata.get("task") in ("robot_lift", "frame_place", "frame_place_ordered"):
        from .robot_lift import asset_hashes
        if asset_hashes() != metadata["robot_assets_sha256"]:
            raise ValueError("Robot assets differ from training")
        dependencies.append(task_path.with_name("robot_lift.py"))
        if metadata.get("task") == "frame_place_ordered":
            dependencies.append(task_path.with_name("task_frame.py"))
    for dependency in dependencies:
        task_hash = hashlib.sha256(dependency.read_bytes()).hexdigest()
        if metadata["source_sha256"][dependency.name] != task_hash:
            raise ValueError("Task source does not match the run")
    splits = task.datasets()
    for split, layouts in splits.items():
        if metadata["splits"][split] != [layout.key for layout in layouts]:
            raise ValueError(f"Split mismatch: {split}")
    precomputed = {}
    if workers < 1:
        raise ValueError('workers must be positive')
    if workers > 1:
        items = []
        for label, rows_key in (("baseline", "before"), ("trained", "after")):
            rows = report[rows_key]
            if [r['layout_id'] for r in rows] != metadata['splits']['test']:
                raise ValueError(f'Missing, reordered, or duplicate cases: {label}')
            items.extend((label, i, row['layout_id'], row['completion'], row['truncated'])
                         for i, row in enumerate(rows))
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_test_worker,
                                 initargs=(metadata.get('task', 'pick_place'),)) as pool:
            precomputed = dict(pool.map(test_replay_worker, items, chunksize=1))
    for label, rows_key in (("baseline", "before"), ("trained", "after")):
        rows = report[rows_key]
        if [r["layout_id"] for r in rows] != metadata["splits"]["test"]:
            raise ValueError(f"Missing, reordered, or duplicate cases: {label}")
        for index, (layout, row) in enumerate(zip(splits["test"], rows)):
            if row["prompt"] != task.prompt(layout):
                raise ValueError("Prompt mismatch")
            replay = (precomputed[(label, index)] if precomputed else
                      task.execute(layout, row["completion"]))
            if row["truncated"]:
                replay.update(reward=0., success=False, valid=False, error="truncated")
            # Normalize tuples/lists through JSON before comparing full traces.
            replay = json.loads(json.dumps(replay))
            keys = ["success", "valid", "error", "trace"]
            if metadata.get("task") in ("frame_place", "frame_place_ordered"):
                keys += ["lifted", "arrived"]
            if metadata.get("task") == "frame_place_ordered":
                keys += ["order_check"]
            if metadata.get("task") in ("robot_lift", "frame_place", "frame_place_ordered"):
                keys += ["final", "components", "time"]
            elif physics:
                keys += ["final_state", "final_distance_m", "final_speed_m_s", "components"]
            for key in keys:
                if (key in replay) != (key in row) or not equivalent(replay.get(key), row.get(key), 1e-7 if metadata.get("task") in ("robot_lift", "frame_place", "frame_place_ordered") else 1e-8 if physics else 0.):
                    raise ValueError(f"Replay mismatch: {label}/{layout.key}/{key}")
            if abs(replay["reward"]-row["reward"]) > 1e-9:
                raise ValueError("Reward mismatch")
        expected = {"n": len(rows), "successes": sum(r["success"] for r in rows),
                    "success_rate": sum(r["success"] for r in rows)/len(rows),
                    "mean_reward": sum(r["reward"] for r in rows)/len(rows),
                    "valid_rate": sum(r["valid"] for r in rows)/len(rows)}
        if any(abs(value-report[label][key]) > 1e-12 for key, value in expected.items()):
            raise ValueError(f"Incorrect aggregate metrics: {label}")
    pairs = list(zip(report["before"], report["after"]))
    wins = sum(not a["success"] and b["success"] for a,b in pairs)
    losses = sum(a["success"] and not b["success"] for a,b in pairs)
    if (wins, losses) != (report["paired_wins"], report["paired_losses"]):
        raise ValueError("Incorrect paired counts")
    delta = (wins-losses)/len(pairs)
    if abs(delta-report["delta_success_rate"]) > 1e-9:
        raise ValueError("Incorrect improvement")
    if "paired_bootstrap_95_ci" in report:
        deltas = [int(b["success"])-int(a["success"]) for a, b in pairs]
        rng = random.Random(20261023)
        boot = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(10000))
        if report["paired_bootstrap_95_ci"] != [boot[250], boot[9749]]:
            raise ValueError("Incorrect paired bootstrap interval")
    return {"audit": "passed", "replayed_programs": 2*len(pairs),
            "paired_wins": wins, "paired_losses": losses,
            "delta_success_rate": delta,
            "test_report_sha256": hashlib.sha256((directory/"test_comparison.json").read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.workers), indent=2))
