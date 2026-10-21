"""Post-hoc geometry/contact audit from raw states, outside the model process.

Uses the run's frozen scoring implementation. Checks logged poses against
reconstructed raw states. Substep peak penetration and support forces remain
logged measurements, not independently reconstructed substep force trajectories.
"""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import mujoco

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    args = p.parse_args()
    folder = Path(args.run).resolve() / "evaluator"
    result = json.loads((folder / "result.json").read_text())
    config = json.loads((folder / "configuration.json").read_text())
    bundle = Path(config["arguments"]["bundle"])
    # Run this auditor from the frozen bundle so all imported scoring dependencies
    # resolve there, even though this standalone auditor is a later analysis file.
    if Path.cwd().resolve() != bundle:
        raise ValueError("Auditor must run with frozen bundle cwd.")
    sys.path.insert(0, str(bundle))
    from experiments.assembly_benchmark.freeze import verify
    from experiments.assembly_benchmark.evaluation import Evaluation
    from experiments.torso_assembly.environment import SIZES
    verify(bundle)
    c = json.loads((bundle / "experiments/assembly_benchmark/protocol.json").read_text())["acceptance"]
    evaluator = Evaluation(c)
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    env = SimpleNamespace(m=model, d=data, woods=[model.body(f"wood{i}").id for i in range(4)],
                          panel=model.body("panel").id, peak=0.)
    raw = np.load(folder / "raw.npz")["rows"]
    continuity = bool(len(raw) and raw[0, 0] <= .0200001 and
                      (len(raw) < 2 or np.max(abs(np.diff(raw[:, 0]) - .02)) < 1e-7))
    max_position_difference = max_rotation_difference = 0.
    last_metrics = None
    count = 0
    with gzip.open(folder / "trace.jsonl.gz", "rt") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if index >= len(raw):
                raise ValueError("Trace longer than raw states.")
            values = raw[index]
            if abs(values[0] - row["time"]) > 1e-8:
                raise ValueError("Trace/raw time mismatch.")
            if row["stage"].startswith("flip:") and evaluator.ref is None:
                if last_metrics is None:
                    raise ValueError("Missing continuous preflip state.")
                env.panel_metrics = lambda: last_metrics
                evaluator.begin_flip(env)
            data.time = values[0]
            data.qpos[:] = values[1:1 + model.nq]
            data.qvel[:] = values[1 + model.nq:1 + model.nq + model.nv]
            data.ctrl[:] = values[1 + model.nq + model.nv:1 + model.nq + model.nv + model.nu]
            mujoco.mj_forward(model, data)
            metrics = row["metrics"]
            bodies = env.woods + [env.panel]
            positions = [x["center"] for x in metrics["frame"]["wood_evaluator_only"]] + [metrics["center_evaluator_only"]]
            rotations = [x["rotation"] for x in metrics["frame"]["wood_evaluator_only"]] + [metrics["rotation_evaluator_only"]]
            for body, position, rotation in zip(bodies, positions, rotations):
                max_position_difference = max(max_position_difference, float(np.max(abs(data.xpos[body] - position))))
                max_rotation_difference = max(max_rotation_difference, float(np.max(abs(data.xmat[body].reshape(3, 3) - rotation))))
            env.peak = metrics["frame"]["peak_penetration_m"]
            env.nails = SimpleNamespace(metrics=lambda: metrics["fasteners"])
            evaluator.update(env, metrics)
            last_metrics = metrics
            count += 1
    if count != len(raw):
        raise ValueError("Trace shorter than raw states.")
    audit = evaluator.result(env, result["finished"], result["flipped"], result["error"])
    audit.update(continuous_raw_time=continuity, reconstructed_pose_difference_m=max_position_difference,
                 reconstructed_rotation_matrix_difference=max_rotation_difference, rows=count,
                 force_and_substep_peaks_source="recorded runtime instrumentation",
                 geometry_and_contacts_source="independent raw-state reconstruction",
                 auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 frozen_manifest_sha256=config["freeze"]["manifest_sha256"],
                 model_result=not result.get("infrastructure_only", False))
    audit["passes"] &= continuity and max_position_difference < 1e-8 and max_rotation_difference < 1e-8
    (folder / "independent_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
