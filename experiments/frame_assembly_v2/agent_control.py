"""Persistent action-only bridge for an external model controlling the hand.

The model receives privileged state and images. It cannot request object poses,
forces, physics edits or assisted grasp constraints through this interface.
Snapshots are internal; replay independently reconstructs the logged actions.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import mujoco
import numpy as np
from PIL import Image

from .environment import FrameAssembly, NAMES, ROOT

BASE = ROOT / "outputs/frame_assembly/astra"
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


class ManipulationAudit:
    """Checks actual contact-supported lifting, rather than wrist motion alone."""
    def __init__(self, env, recorder):
        self.env, self.recorder = env, recorder
        self.initial = {n: env.data.xpos[b].copy() for n, b in env.bids.items()}
        self.corners = np.array(list(itertools.product((-1, 1), repeat=3)))
        self.hold = {n: 0. for n in NAMES}
        self.stable_carry = {n: 0. for n in NAMES}
        self.last_relative_pose = {}
        self.results = {n: {"max_center_displacement_m": 0., "max_lowest_point_above_table_m": 0., "longest_airborne_with_hand_contact_seconds": 0., "longest_stable_carried_seconds": 0., "ever_hand_contact": False} for n in NAMES}
        self.max_penetration = 0.

    def add(self, phase):
        self.recorder.add(phase)
        e, d, m = self.env, self.env.data, self.env.model
        hand_force = {n: 0. for n in NAMES}
        other_force = {n: 0. for n in NAMES}
        penetration = {n: 0. for n in NAMES}
        reverse = {b: n for n, b in e.bids.items()}
        f = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = (int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            mujoco.mj_contactForce(m, d, i, f)
            for board, other in ((a, b), (b, a)):
                if board in reverse:
                    n = reverse[board]
                    self.max_penetration = max(self.max_penetration, -float(c.dist))
                    penetration[n] = max(penetration[n], -float(c.dist))
                    target = hand_force if other in e.hand_body_ids else other_force
                    target[n] += max(0., float(f[0]))
        for n, bid in e.bids.items():
            r = self.results[n]
            heights = []
            for part in ("slab", "shoulder"):
                gid = m.geom(f"{n}_{part}").id
                world = (self.corners * m.geom_size[gid]) @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
                heights.append(float(world[:, 2].min()))
            lowest = min(heights)
            r["max_lowest_point_above_table_m"] = max(r["max_lowest_point_above_table_m"], lowest)
            r["max_center_displacement_m"] = max(r["max_center_displacement_m"], float(np.linalg.norm(d.xpos[bid] - self.initial[n])))
            r["ever_hand_contact"] |= hand_force[n] > .01
            airborne = lowest > .003 and hand_force[n] > .01 and other_force[n] < .01
            self.hold[n] = self.hold[n] + e.dt if airborne else 0.
            r["longest_airborne_with_hand_contact_seconds"] = max(r["longest_airborne_with_hand_contact_seconds"], self.hold[n])
            wrist = m.body("r_mount").id
            rotation = d.xmat[wrist].reshape(3, 3)
            relative_position = rotation.T @ (d.xpos[bid] - d.xpos[wrist])
            relative_rotation = rotation.T @ d.xmat[bid].reshape(3, 3)
            stable = False
            if n in self.last_relative_pose:
                old_p, old_r = self.last_relative_pose[n]
                slip_speed = np.linalg.norm(relative_position - old_p) / e.dt
                spin_speed = np.arccos(np.clip((np.trace(old_r.T @ relative_rotation) - 1) / 2, -1, 1)) / e.dt
                stable = airborne and slip_speed < .03 and spin_speed < .25 and penetration[n] < .001
            self.last_relative_pose[n] = (relative_position.copy(), relative_rotation.copy())
            self.stable_carry[n] = self.stable_carry[n] + e.dt if stable else 0.
            r["longest_stable_carried_seconds"] = max(r["longest_stable_carried_seconds"], self.stable_carry[n])


def validate(action):
    if not isinstance(action, dict):
        raise ValueError("Action must be a JSON object")
    extra = set(action) - {"seconds", "wrist_position", "wrist_quaternion", "finger_angles", "reason"}
    if extra:
        raise ValueError(f"Unsupported action fields: {sorted(extra)}")
    seconds = float(action.get("seconds", .5))
    if not np.isfinite(seconds) or not .02 <= seconds <= 3:
        raise ValueError("seconds must be between 0.02 and 3.0")
    result = {"seconds": round(seconds / .02) * .02, "reason": str(action.get("reason", ""))[:500]}
    for field, shape in (("wrist_position", 3), ("wrist_quaternion", 4), ("finger_angles", 20)):
        if field in action:
            a = np.array(action[field], dtype=float)
            if a.shape != (shape,) or not np.isfinite(a).all():
                raise ValueError(f"{field} must contain {shape} finite numbers")
            if field == "wrist_quaternion" and np.linalg.norm(a) < 1e-8:
                raise ValueError("Quaternion cannot be zero")
            if field == "wrist_position" and (np.any(a < [-.60, -.47, .04]) or np.any(a > [.60, .47, .75])):
                raise ValueError("Wrist target outside the workspace bounds")
            result[field] = a.tolist()
    return result


def apply(env, action, recorder=None):
    params = {k: v for k, v in action.items() if k not in ("seconds", "reason")}
    for _ in range(round(action["seconds"] / env.dt)):
        if any(env.data.eq_active[e] for e in env.carries.values()):
            raise RuntimeError("Board assistance unexpectedly active")
        env.step(**params)
        if recorder:
            recorder.add(action["reason"] or "Astra wrist and finger action")


def snapshot(env, path):
    state = np.empty(mujoco.mj_stateSize(env.model, STATE))
    mujoco.mj_getState(env.model, env.data, state, STATE)
    np.savez(path / "checkpoint.npz", integration_state=state, stable_seconds=env.stable_seconds)


def load(path):
    meta = json.loads((path / "session.json").read_text())
    env = FrameAssembly(allow_assistance=False)
    if hashlib.sha256(env.xml.encode()).hexdigest() != meta["scene_sha256"]:
        raise RuntimeError("Scene changed since initialization; start a new trial explicitly")
    state = np.load(path / "checkpoint.npz")
    mujoco.mj_setState(env.model, env.data, state["integration_state"], STATE)
    env.stable_seconds = float(state["stable_seconds"])
    mujoco.mj_forward(env.model, env.data)
    if any(env.data.eq_active[e] for e in env.carries.values()):
        raise RuntimeError("Assistance is forbidden in the control experiment")
    return env, meta


def observation(env, path, index, render=True):
    m, d = env.model, env.data
    contact_forces = {}
    force = np.zeros(6)
    hand_contacts = []
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = (int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2))
        if a not in env.hand_body_ids and b not in env.hand_body_ids:
            continue
        mujoco.mj_contactForce(m, d, i, force)
        if force[0] <= .001:
            continue
        key = f"{m.body(a).name or 'world'} / {m.body(b).name or 'world'}"
        contact_forces[key] = contact_forces.get(key, 0.) + float(force[0])
        if a in env.bids.values() or b in env.bids.values():
            hand_contacts.append(key)
    obs = {
        "time": float(d.time), "wrist_position": d.xpos[m.body("r_mount").id].tolist(),
        "wrist_quaternion": d.xquat[m.body("r_mount").id].tolist(),
        "wrist_target": d.mocap_pos[env.wrist_id].tolist(),
        "fingertip_positions": {m.site(i).name: d.site_xpos[i].tolist() for i in range(m.nsite) if m.site(i).name.endswith("_tip")},
        "joint_order": [m.joint(int(i)).name for i in env.joint_ids],
        "finger_angles": d.qpos[env.finger_qadr].tolist(), "finger_targets": d.ctrl.tolist(),
        "joint_limits": m.actuator_ctrlrange.tolist(),
        "board_poses": {n: np.r_[d.xpos[b], d.xquat[b]].tolist() for n, b in env.bids.items()},
        "goals": {n: v.tolist() for n, v in env.goal.items()},
        "contact_normal_forces_N": contact_forces, "hand_board_contacts": sorted(set(hand_contacts)),
        "metrics": env.metrics(), "warnings": [int(w.number) for w in d.warning],
    }
    if render:
        option = mujoco.MjvOption()
        option.geomgroup[2] = 0
        option.sitegroup[:] = 0
        camera = mujoco.MjvCamera()
        with mujoco.Renderer(m, height=600, width=800) as renderer:
            for name in ("overview", "hand"):
                camera.lookat[:] = [0, 0, .07] if name == "overview" else d.xpos[m.body("r_mount").id] + [0, 0, -.12]
                camera.distance = 1.38 if name == "overview" else .52
                camera.azimuth, camera.elevation = 115, -35 if name == "hand" else -49
                renderer.update_scene(d, camera=camera, scene_option=option)
                target = path / f"{name}_{index:03d}.png"
                Image.fromarray(renderer.render()).save(target)
                obs[f"{name}_image"] = str(target)
    (path / f"observation_{index:03d}.json").write_text(json.dumps(obs, indent=2))
    return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("init", "observe", "act", "replay"))
    parser.add_argument("--trial", default="trial01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="Action JSON; see validate() for schema")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.trial):
        raise ValueError("Invalid trial name")
    path = BASE / args.trial
    if args.operation == "init":
        path.mkdir(parents=True, exist_ok=False)
        env = FrameAssembly(allow_assistance=False)
        env.reset(seed=args.seed)
        env.save_scene(path)
        meta = {"seed": args.seed, "controller": "gpt-6-astra delegated agent", "mode": "model-selected action chunks via tool calls; not real-time token-to-action inference", "observation": "images and privileged simulator state", "board_assistance": False, "virtual_wrist": True, "scene_sha256": hashlib.sha256(env.xml.encode()).hexdigest(), "started_unix": time.time()}
        (path / "session.json").write_text(json.dumps(meta, indent=2))
        (path / "actions.jsonl").write_text("")
        snapshot(env, path)
        print(json.dumps(observation(env, path, 0, not args.no_render)))
        return
    env, meta = load(path)
    lines = (path / "actions.jsonl").read_text().splitlines()
    index = len(lines)
    if args.operation == "act":
        action = validate(json.loads(args.json))
        before = env.data.time
        apply(env, action)
        snapshot(env, path)
        with (path / "actions.jsonl").open("a") as f:
            f.write(json.dumps({"index": index + 1, "start_sim_time": before, "end_sim_time": env.data.time, "wall_unix": time.time(), "action": action}) + "\n")
        index += 1
    elif args.operation == "replay":
        from .run import Recorder
        expected = env.data.qpos.copy()
        env.reset(seed=meta["seed"])
        export = path / "replay"
        export.mkdir(exist_ok=True)
        recorder = Recorder(env, export, video=not args.no_render, subtitle="Astra-controlled wrist + fingers | physical contacts | board assistance OFF")
        audit = ManipulationAudit(env, recorder)
        try:
            for line in lines:
                apply(env, validate(json.loads(line)["action"]), audit)
        finally:
            if recorder.rows:
                recorder.close()
        difference = float(np.max(np.abs(expected - env.data.qpos)))
        summary = {**meta, "actions": len(lines), "sim_seconds": env.data.time, "assembly": env.metrics(), "manipulation": audit.results, "max_board_contact_penetration_during_episode_m": audit.max_penetration, "airborne_criterion": "Every rail geometry corner >3 mm above table, positive hand contact >0.01 N, total non-hand contact <0.01 N; contact does not by itself imply a stable grasp", "stable_carry_criterion": "Airborne plus relative-to-wrist translation speed <0.03 m/s, rotation speed <0.25 rad/s, and board contact penetration <1 mm; evaluated at 50 Hz", "replay_max_qpos_difference": difference, "replay_matches": difference < 1e-5, "warnings": [int(w.number) for w in env.data.warning]}
        (path / "result.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return
    print(json.dumps(observation(env, path, index, not args.no_render)))


if __name__ == "__main__":
    main()
