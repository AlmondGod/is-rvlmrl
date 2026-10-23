"""Execute generated programs in the unchanged three-arm assembly world.

Programs are parsed, never eval/exec'd. The language model supplies the whole
program before simulation starts; the inherited skills use RGB and encoders.
This separate offline code-policy experiment is not the historical live VLM test.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from experiments.assembly_benchmark.interface import INDEXED, SKILLS

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'experiments/assembly_benchmark/protocol.json'
SETTLE_SECONDS = 6.0
MAX_CALLS = 32


def parse_program(text):
    if len(text) > 8192:
        raise ValueError('program_too_long')
    source = text.strip()
    if source.startswith('```'):
        lines = source.splitlines()
        if lines[0] not in ('```', '```python') or lines[-1] != '```':
            raise ValueError('invalid_fence')
        source = '\n'.join(lines[1:-1])
    tree = ast.parse(source)
    if not 1 <= len(tree.body) <= MAX_CALLS:
        raise ValueError('call_budget')
    calls = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ValueError('only_api_calls')
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.func.id not in SKILLS or call.keywords:
            raise ValueError('unknown_api')
        name = call.func.id
        if len(call.args) != (1 if name in INDEXED else 0):
            raise ValueError('argument_count')
        index = None
        if call.args:
            node = call.args[0]
            if not isinstance(node, ast.Constant) or type(node.value) is not int or not 0 <= node.value < 4:
                raise ValueError('literal_index_0_to_3')
            index = node.value
        calls.append((name, index))
    # Do not silently ignore trailing code after a terminal instruction.
    if any(name in ('finish', 'stop') for name, _ in calls[:-1]):
        raise ValueError('code_after_terminal')
    return calls


def provenance():
    """Hash every dependency, including skills, physics, assets and this adapter."""
    from experiments.assembly_benchmark.freeze import ASSETS, PACKAGES
    paths = []
    for directory in [*(ROOT / 'experiments' / p for p in (*PACKAGES, 'cap_rl')),
                      *(ROOT / a for a in ASSETS)]:
        paths.extend(p for p in directory.rglob('*') if p.is_file() and
                     '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo'))
    return {
        'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(set(paths))},
        'versions': {p: importlib.metadata.version(p) for p in
                     ('mujoco', 'numpy', 'scipy', 'opencv-python', 'imageio')},
        'python': platform.python_version(), 'platform': platform.platform(),
        'settle_seconds': SETTLE_SECONDS,
    }


def score(released_members, intact_nails, preflip_stable, inverted_hold, success, unsafe):
    if unsafe:
        return 0.0
    return (.30 * len(set(released_members)) / 4 + .25 * min(intact_nails, 8) / 8 +
            .10 * min(preflip_stable / 3., 1.) + .15 * min(inverted_hold / 5., 1.) +
            .20 * bool(success))


def run(seed, text, out, *, video=False, seconds=1200., progress=False):
    """One continuous simulation. No reset, restore, repair, or oracle control."""
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from experiments.arm_corner.run import clean
    from experiments.assembly_benchmark.evaluation import Evaluation
    from experiments.assembly_benchmark.executor import Executor, update_frame_observations
    from experiments.assembly_benchmark.run import robot
    from experiments.torso_assembly.environment import Env
    from experiments.torso_assembly.frame_vision import MultiVision, probe_anchor
    from experiments.wall_panel.perception import Vision as PanelVision

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'program.py').write_text(text)
    (out / 'provenance.json').write_text(json.dumps(provenance(), indent=2))
    try:
        calls = parse_program(text)
    except (ValueError, SyntaxError) as exc:
        result = dict(success=False, valid=False, reward=0., error=str(exc),
                      seed=seed, simulated_s=0., history=[], physics_started=False)
        (out / 'result.json').write_text(json.dumps(result, indent=2))
        return result
    protocol = json.loads(PROTOCOL.read_text())
    config = protocol['environment']
    env = Env(dt=config['dt'], layout_seed=seed, physical_nails=True,
              fixture_half_width=config['fixture_half_width'],
              shoulder_half_span=config['shoulder_half_span'])
    env.flip_enabled = True
    env.flip_options = dict(grasp='stack', smooth=True, clearance=True)
    executor = Executor(env)
    fv, pv = MultiVision(), PanelVision()
    renderer = mujoco.Renderer(env.m, config['image_height'], config['image_width'])
    evaluation = Evaluation(protocol['acceptance'])
    writer = imageio.get_writer(out / 'episode.mp4', fps=10, codec='libx264', quality=7) if video else None
    (out / 'scene.xml').write_text(env.xml)
    history, samples, released = [], [], set()
    estimates, sheet = [None] * 4, None
    active = None
    next_call = 0
    ready_at = SETTLE_SECONDS
    finished = False
    error = None
    lastphase = None
    started = time.monotonic()
    try:
        for step in range(round(seconds / .02)):
            r = robot(env)
            if step % 5 == 0:
                frames, cameras = env.images(renderer)
                if update_frame_observations(executor):
                    estimates = fv.update(frames, cameras, r, executor.frame.holding,
                                          probe_anchor(executor.frame))
                if r['shuttle_x'] < .03:
                    sheet = pv.update(frames, cameras)
                if writer is not None:
                    renderer.update_scene(env.d, camera='torso_context')
                    writer.append_data(renderer.render().copy())
                samples.append(np.r_[env.d.time, env.d.qpos, env.d.qvel, env.d.ctrl])
            # Fixed settling time is common to every generated program, not
            # model latency. Controllers and gravity continue during this time.
            if active is None and step % 5 == 0 and env.d.time + 1e-8 >= ready_at:
                if next_call == len(calls):
                    error = 'missing_finish'
                    break
                name, index = calls[next_call]
                history.append(dict(skill=name, index=index, started_s=float(env.d.time), status='started'))
                next_call += 1
                if name in ('finish', 'stop'):
                    finished = name == 'finish'
                    history[-1]['status'] = name
                    break
                executor.start(dict(skill=name, index=index), r, estimates, sheet)
                if name == 'flip_structure':
                    evaluation.begin_flip(env)
                active = len(history) - 1
            command = executor.command(r, estimates, sheet)
            completed = active is not None and executor.kind is None
            env.step(*command)
            metrics = env.panel_metrics()
            evaluation.update(env, metrics)
            if completed:
                entry = history[active]
                entry.update(status='completed', completed_s=float(env.d.time),
                             metrics=clean(metrics), fasteners=clean(env.nails.metrics()))
                if entry['skill'] == 'release_member':
                    i = entry['index']
                    f = metrics['frame']
                    # Shaping is measured after physical release, never awarded
                    # solely because a skill reported completion.
                    pose = env.d.xpos[env.woods[i]]
                    from experiments.torso_assembly.environment import GOALS
                    if (f['position_error_m'][i] < .006 and f['yaw_error_deg'][i] < 1. and
                        f['tilt_deg'][i] < 1. and abs(pose[2] - GOALS[i, 2]) < .003 and
                        f['grip_force'][i] < .1):
                        released.add(i)
                active = None
                ready_at = env.d.time + SETTLE_SECONDS
            if progress and executor.phase != lastphase:
                print(f'{env.d.time:.2f} {executor.phase}', flush=True)
                lastphase = executor.phase
            if env.peak > protocol['acceptance']['max_penetration_m'] or evaluation.nonfinite_or_warning:
                raise RuntimeError('numerical_gate')
        else:
            error = 'simulation_deadline'
    except KeyboardInterrupt:
        error = 'interrupted'
    except Exception as exc:
        error = type(exc).__name__ + ':' + str(exc)
        if history and history[-1]['status'] == 'started':
            history[-1].update(status='failed', error=error)
    finally:
        renderer.close()
        if writer is not None:
            writer.close()
    result = evaluation.result(env, finished, executor.flipped, error)
    unsafe = (env.peak > protocol['acceptance']['max_penetration_m'] or
              evaluation.nonfinite_or_warning or bool(evaluation.unexpected))
    result.update(success=result['passes'], valid=True, error=error, seed=seed,
                  reward=score(released, env.nails.metrics()['intact'], evaluation.preflip_stable,
                               evaluation.hold, result['passes'], unsafe),
                  verified_released_members=sorted(released), history=history,
                  fasteners=clean(env.nails.metrics()), finished=finished, flipped=executor.flipped,
                  simulated_s=float(env.d.time), wall_s=time.monotonic()-started,
                  physics_started=True, physics_paused_for_inference=False,
                  generated_before_simulation=True, physical_validation=False,
                  full_visual_language_policy=False, final_state=np.r_[env.d.qpos, env.d.qvel, env.d.ctrl].tolist())
    np.savez_compressed(out / 'states.npz', rows=samples, nq=env.m.nq, nv=env.m.nv, nu=env.m.nu)
    (out / 'result.json').write_text(json.dumps(clean(result), indent=2))
    return clean(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--seconds', type=float, default=1200., help='Development timeout; full evaluation uses 1200.')
    args = parser.parse_args()
    result = run(args.seed, Path(args.program).read_text(), args.out,
                 video=args.video, seconds=args.seconds, progress=True)
    print(json.dumps({k: v for k, v in result.items() if k not in ('history', 'final_state', 'last_flip')}, indent=2))


if __name__ == '__main__':
    main()
