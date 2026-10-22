"""Replay saved test programs and independently verify the reported improvement.

This audit needs only the Python standard library, not a GPU or model weights.
It verifies execution and accounting, not whether the claimed model emitted text.
Model-generation provenance is provided by the checkpoint and saved source/config.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from .tasks import get_task


def audit(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    report = json.loads((directory / "test_comparison.json").read_text())
    task = get_task(metadata.get("task", "pick_place"))
    task_path = Path(task.__file__)
    dependencies = [task_path]
    if metadata.get("task") == "ordered_two":
        dependencies.append(task_path.with_name("task.py"))  # shared safe grammar
    for dependency in dependencies:
        task_hash = hashlib.sha256(dependency.read_bytes()).hexdigest()
        if metadata["source_sha256"][dependency.name] != task_hash:
            raise ValueError("Task source does not match the run")
    splits = task.datasets()
    for split, layouts in splits.items():
        if metadata["splits"][split] != [layout.key for layout in layouts]:
            raise ValueError(f"Split mismatch: {split}")
    for label, rows_key in (("baseline", "before"), ("trained", "after")):
        rows = report[rows_key]
        if [r["layout_id"] for r in rows] != metadata["splits"]["test"]:
            raise ValueError(f"Missing, reordered, or duplicate cases: {label}")
        for layout, row in zip(splits["test"], rows):
            if row["prompt"] != task.prompt(layout):
                raise ValueError("Prompt mismatch")
            replay = task.execute(layout, row["completion"])
            if row["truncated"]:
                replay.update(reward=0., success=False, valid=False, error="truncated")
            # Normalize tuples/lists through JSON before comparing full traces.
            replay = json.loads(json.dumps(replay))
            for key in ("success", "valid", "error", "trace"):
                if replay[key] != row[key]:
                    raise ValueError(f"Replay mismatch: {label}/{layout.key}/{key}")
            if abs(replay["reward"]-row["reward"]) > 1e-9:
                raise ValueError("Reward mismatch")
        expected = {"n": len(rows), "successes": sum(r["success"] for r in rows),
                    "success_rate": sum(r["success"] for r in rows)/len(rows),
                    "mean_reward": sum(r["reward"] for r in rows)/len(rows),
                    "valid_rate": sum(r["valid"] for r in rows)/len(rows)}
        if expected != report[label]:
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
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
