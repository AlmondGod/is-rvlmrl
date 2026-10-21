"""Matched model benchmark against frozen high-level feedback skills.

Run from a verified bundle, with outputs outside it. Both models receive the same
prompt, image encoding, camera selection and packet schema. Realized observations
can differ after their decisions/latencies diverge; paired initial conditions match.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
import traceback

import imageio.v2 as imageio
import mujoco
import numpy as np

from experiments.arm_corner.run import clean
from experiments.torso_assembly.environment import Env, GOALS, SIZES, PANEL_GOAL, PANEL_SIZE
from experiments.torso_assembly.frame_vision import MultiVision, probe_anchor
from experiments.wall_panel.perception import Vision as PanelVision
from .executor import Executor, update_frame_observations
from .evaluation import Evaluation
from .freeze import verify
from .interface import public_robot, public_events, validate_action, response_ready
from .provider import query

def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2))

def robot(env):
    result = env.robot()
    for name in ("shuttle_x", "rack_slide0", "rack_slide1"):
        result[name] = float(env.d.qpos[int(env.m.joint(name).qposadr[0])])
    return result

def encode_images(frames):
    result = {}
    for k in (0, 2, 3):
        buffer = io.BytesIO()
        imageio.imwrite(buffer, frames[k], format="png")
        result["camera" + str(k)] = buffer.getvalue()
    return result

def packet(env, executor, cameras, history, index):
    return clean({
        "observation_index": index,
        "robot": public_robot(robot(env)), "phase": executor.phase,
        "last_failure": executor.failure,
        "camera_calibration": {str(i): cameras[i] for i in (0, 2, 3)},
        "blueprint": {"centers_m": GOALS, "member_sizes_m": SIZES,
                      "panel_size_m": PANEL_SIZE, "panel_goal_m": PANEL_GOAL},
        "tool_triggers": public_events(executor.events),
        "command_history": history
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=("qwen", "astra"), required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--backend", choices=("chat", "responses", "codex"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--revision", required=True, help="Resolved weights revision or provider model version; never guess.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--development-smoke", action="store_true")
    parser.add_argument("--harness-test", action="store_true",
                        help="Explicit scripted/mock transport test; never a model result.")
    parser.add_argument("--seconds", type=float, help="Development smoke only.")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    if Path.cwd().resolve() != bundle:
        raise ValueError("Run from frozen bundle cwd to resolve frozen imports/assets.")
    frozen = verify(bundle)
    if not Path(__file__).resolve().is_relative_to(bundle):
        raise RuntimeError("Not importing frozen runtime.")
    protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
    if args.seed not in protocol["paired_evaluation_seeds"] and not args.development_smoke:
        raise ValueError("Seed outside frozen evaluation set.")
    if args.development_smoke and args.seed != protocol["development_seed"]:
        raise ValueError("Development uses the development seed only.")
    if args.seconds and not args.development_smoke:
        raise ValueError("Cannot change evaluation time budget.")
    if args.harness_test and not args.development_smoke:
        raise ValueError("Harness tests must be development-only.")
    if args.model == "qwen" and not args.harness_test and args.revision != protocol["qwen_weight_revision"]:
        raise ValueError("Qwen revision differs from frozen checkpoint.")
    if protocol["no_runpod"] and "runpod" in args.base_url.lower():
        raise ValueError("RunPod prohibited.")
    # Fail credential/config checks before starting a costly physics trial.
    import os
    if args.api_key_env and not os.environ.get(args.api_key_env):
        raise RuntimeError("Missing credential: " + args.api_key_env)
    out = Path(args.out).resolve()
    if out.is_relative_to(bundle):
        raise ValueError("Outputs must be outside frozen bundle.")
    out.mkdir(parents=True, exist_ok=False)
    private = out / "evaluator"
    private.mkdir()
    public = out / "observations"
    public.mkdir()
    econf = protocol["environment"]
    env = Env(dt=econf["dt"], layout_seed=args.seed, physical_nails=True,
              fixture_half_width=econf["fixture_half_width"],
              shoulder_half_span=econf["shoulder_half_span"])
    env.flip_enabled = True
    env.flip_options = {"grasp": "stack", "smooth": True, "clearance": True}
    executor = Executor(env)
    fv, pv = MultiVision(), PanelVision()
    renderer = mujoco.Renderer(env.m, econf["image_height"], econf["image_width"])
    evaluation = Evaluation(protocol["acceptance"])
    save(private / "configuration.json", {"arguments": vars(args), "freeze": frozen,
         "model_id": protocol["models"][0 if args.model == "qwen" else 1],
         "benchmark_scope": protocol["scope"], "model_settings": protocol["model_settings"][args.model]})
    (private / "scene.xml").write_text(env.xml)
    trace = gzip.open(private / "trace.jsonl.gz", "wt")
    writer = imageio.get_writer(out / "continuous.mp4", fps=10, codec="libx264", quality=7)
    pool = ThreadPoolExecutor(max_workers=1)
    future = response = None
    request_time = 0.
    history, calls, raw, control_stalls = [], [], [], []
    estimates, sheet, frames, cameras = [None] * 4, None, None, None
    active = None
    finished = False
    error = failure_class = None
    maxlag = 0.
    started = time.monotonic()
    decision = 0
    lastphase = None
    budget = protocol["budgets"]
    seconds = args.seconds or budget["simulated_seconds"]
    try:
        for step in range(round(seconds / .02)):
            r = robot(env)
            if step % 5 == 0:
                frames, cameras = env.images(renderer)
                if update_frame_observations(executor):
                    estimates = fv.update(frames, cameras, r, executor.frame.holding,
                                          probe_anchor(executor.frame))
                if r["shuttle_x"] < .03:
                    sheet = pv.update(frames, cameras)
                renderer.update_scene(env.d, camera="torso_overview")
                writer.append_data(renderer.render().copy())
            if future is not None and future.done() and response is None:
                try:
                    response = future.result()
                except Exception as exc:
                    response = {"error": type(exc).__name__ + ":" + str(exc),
                                "latency_wall_s": time.monotonic() - request_wall}
                save(private / f"response_{decision - 1:03d}.json", response)
                calls.append({"call": decision - 1, "observation_sim_s": request_time,
                              "latency_wall_s": response["latency_wall_s"],
                              "request_sha256": response.get("request_sha256"),
                              "error": response.get("error"),
                              "returned_model": response.get("response", {}).get("model"),
                              "response_id": response.get("response", {}).get("id"),
                              "usage": response.get("response", {}).get("usage")})
            if response is not None and response_ready(request_time, response["latency_wall_s"], env.d.time):
                calls[-1]["applied_sim_s"] = float(env.d.time)
                if response.get("error"):
                    failure_class = "inference_infrastructure"
                    raise RuntimeError(response["error"])
                try:
                    command = validate_action(response["text"])
                except Exception:
                    failure_class = "model_action_format"
                    raise
                history.append({"command": command, "issued_sim_s": float(env.d.time), "status": "started"})
                response = future = None
                if command["skill"] in ("finish", "stop"):
                    finished = command["skill"] == "finish"
                    history[-1]["status"] = "finished" if finished else "stopped"
                    break
                try:
                    before = time.monotonic()
                    executor.start(command, r, estimates, sheet)
                    duration = time.monotonic() - before
                    control_stalls.append({"time_s": float(env.d.time), "kind": "start_skill", "wall_s": duration})
                    if command["skill"] == "flip_structure":
                        evaluation.begin_flip(env)
                except Exception:
                    failure_class = "skill_precondition_or_planning"
                    raise
                active = len(history) - 1
            # Publish at a fresh camera tick, only when an action can be selected.
            if future is None and executor.kind is None and step % 5 == 0:
                if decision >= budget["model_calls"]:
                    failure_class = "model_call_budget"
                    raise RuntimeError("model_call_budget")
                public_packet = packet(env, executor, cameras, history, decision)
                images = encode_images(frames)
                obs = public / f"{decision:03d}"
                obs.mkdir()
                for name, data in images.items():
                    (obs / (name + ".png")).write_bytes(data)
                save(obs / "packet.json", public_packet)
                save(obs / "image_hashes.json", {k: hashlib.sha256(v).hexdigest() for k, v in images.items()})
                request_time = float(env.d.time)
                request_wall = time.monotonic()
                future = pool.submit(query, model=args.served_model, backend=args.backend,
                    base_url=args.base_url, api_key_env=args.api_key_env, packet=public_packet,
                    images=images, settings=protocol["model_settings"][args.model],
                    max_tokens=budget["max_output_tokens_per_call"],
                    timeout=budget["request_timeout_wall_s"])
                decision += 1
            before = time.monotonic()
            command = executor.command(r, estimates, sheet)
            duration = time.monotonic() - before
            if duration > .02:
                control_stalls.append({"time_s": float(env.d.time), "kind": "control_tick", "wall_s": duration})
            if active is not None and executor.kind is None:
                history[active].update(status="completed", completed_sim_s=float(env.d.time))
                active = None
            env.step(*command)
            metrics = env.panel_metrics()
            evaluation.update(env, metrics)
            metrics["fasteners"] = env.nails.metrics()
            row = {"time": float(env.d.time), "stage": executor.phase, "robot": clean(robot(env)),
                   "estimates": clean(estimates), "sheet": clean(sheet),
                   "vision_diagnostics": clean(fv.diagnostics), "metrics": metrics,
                   "command": clean(command)}
            trace.write(json.dumps(row, separators=(",", ":")) + "\n")
            raw.append(np.r_[env.d.time, env.d.qpos, env.d.qvel, env.d.ctrl, env.d.actuator_force])
            if executor.phase != lastphase:
                print(f"{env.d.time:.2f} {executor.phase}", flush=True)
                lastphase = executor.phase
            if env.peak > protocol["acceptance"]["max_penetration_m"] or evaluation.nonfinite_or_warning:
                failure_class = "numerical"
                raise RuntimeError("numerical_gate")
            if time.monotonic() - started > budget["trial_wall_seconds"]:
                failure_class = "wall_budget"
                raise RuntimeError("wall_budget")
            delay = started + (step + 1) * .02 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                maxlag = max(maxlag, -delay)
        else:
            error = "simulation_deadline"
            failure_class = "timeout"
    except KeyboardInterrupt:
        error = "interrupted"
        failure_class = "interrupted"
    except Exception as exc:
        error = repr(exc)
        failure_class = failure_class or "skill_execution"
        (private / "exception.txt").write_text(traceback.format_exc())
    finally:
        trace.close()
        writer.close()
        renderer.close()
        # In-flight inference cannot affect a closed simulation.
        pool.shutdown(wait=False, cancel_futures=True)
        np.savez_compressed(private / "raw.npz", rows=raw, nq=env.m.nq, nv=env.m.nv, nu=env.m.nu)
        result = evaluation.result(env, finished, executor.flipped, error)
        if error is None and not result["passes"]:
            failure_class = "acceptance_failure_after_finish" if finished else "model_stop"
        result.update(error=error, failure_class=failure_class, finished=finished,
            flipped=executor.flipped, model=None if args.harness_test else
            ("Codex(" + args.served_model + ")" if args.backend == "codex" else args.served_model),
            model_route=args.backend,
            model_revision=args.revision, infrastructure_only=args.harness_test,
            verified_responses=len(calls), actual_astra=None if args.backend == "codex" else
            (not args.harness_test and args.model == "astra" and
             any(c.get("returned_model") == args.served_model for c in calls)),
            model_identity_evidence="CLI configured; provider snapshot unverified" if args.backend == "codex"
            else "requested and returned API model IDs recorded",
            development_smoke=args.development_smoke, seed_evaluator_only=args.seed,
            continuous=True, simulated_s=float(env.d.time), wall_s=time.monotonic() - started,
            max_wall_lag_s=maxlag, synchronous_control_stalls=control_stalls,
            physics_paused_for_inference=False, real_time_deployment_validated=False,
            events=executor.events, calls=calls, history=history,
            fasteners=env.nails.metrics(), physical_validation=False,
            no_runpod=protocol["no_runpod"])
        save(private / "result.json", result)
        print(json.dumps({k: result[k] for k in ("passes", "error", "failure_class", "simulated_s", "wall_s")}), flush=True)

if __name__ == "__main__":
    main()
