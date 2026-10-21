"""Audit development scene, not evaluation seeds; saves evaluator-only evidence."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np

from experiments.arm_corner.run import clean
from experiments.torso_assembly.environment import Env
from experiments.torso_assembly.fasteners import NailParameters
from experiments.torso_assembly.frame_vision import MultiVision
from .executor import Executor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-scene", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    env = Env(dt=.0005, layout_seed=3, physical_nails=True,
              fixture_half_width=.38, shoulder_half_span=.55)
    reference = mujoco.MjModel.from_xml_path(args.reference_scene)
    fields = ("body_mass", "body_inertia", "body_pos", "geom_size", "geom_friction",
              "geom_solref", "geom_solimp", "jnt_range", "actuator_gainprm",
              "actuator_biasprm", "actuator_forcerange", "pair_friction",
              "pair_solref", "pair_solimp", "eq_type", "eq_obj1id", "eq_obj2id",
              "qpos0", "cam_pos", "cam_quat", "cam_fovy", "mesh_vert", "mesh_face")
    equivalence = {name: bool(np.array_equal(getattr(env.m, name), getattr(reference, name)))
                   for name in fields}
    renderer = mujoco.Renderer(env.m, 960, 1280)
    frames, cameras = env.images(renderer)
    vision = MultiVision()
    estimates = vision.update(frames, cameras, env.robot())
    for i in (0, 2, 3):
        imageio.imwrite(out / f"camera{i}.png", frames[i])
    renderer.update_scene(env.d, camera="torso_overview")
    imageio.imwrite(out / "overview.png", renderer.render())
    renderer.close()
    # Raw measurements are private evaluator evidence, never actor inputs.
    errors = [None if e is None else float(np.linalg.norm(e["center"] - env.d.xpos[b]))
              for e, b in zip(estimates, env.woods)]
    executor = Executor(env)
    start = time.monotonic()
    for _ in range(50):
        env.step(*executor.cmd)
    output = {
        "development_seed": 3, "full_task_feasibility_reference": "physical_nails_flip83",
        "compiled_model_matches_previous_success": equivalence,
        "dt_s": float(env.m.opt.timestep), "native_arm_count": 3,
        "workpiece_masses_kg": env.m.body_mass[env.woods + [env.panel]].tolist(),
        "actuator_force_limits": env.m.actuator_forcerange.tolist(),
        "nail_parameters": asdict(NailParameters()),
        "initial_camera_estimate_errors_m_evaluator_only": errors,
        "idle_one_sim_second_wall_s": time.monotonic() - start,
        "idle_peak_penetration_m": env.peak,
        "idle_solver_warnings": env.d.warning.number.tolist(),
        "physical_validation": False,
        "scope": "Compiled geometry/dynamics equality and one-second idle smoke; not robustness or physical validation."
    }
    (out / "scene.xml").write_text(env.xml)
    (out / "camera_calibration.json").write_text(json.dumps(clean(cameras), indent=2))
    (out / "audit.json").write_text(json.dumps(clean(output), indent=2))
    print(json.dumps(clean(output), indent=2))

if __name__ == "__main__":
    main()
