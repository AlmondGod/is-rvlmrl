"""Two-station inertial docking, executed by MuJoCo rather than a grid update.

A 1 kg planar slider has two force motors, zero friction/damping and no automatic
braking. All trajectory state comes from mj_step. No teleporting or target servo.
"""
import ast
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import random

import mujoco
import numpy as np

SYSTEM = 'Write only the requested Python API program. Do not explain it.'
DT = .005
PULSE_STEPS = 100
DOCK_STEPS = 50
POSITION_TOLERANCE = .015
SPEED_TOLERANCE = .025
XML = '''<mujoco model="inertial_docking">
  <option timestep="0.005" gravity="0 0 0" integrator="RK4"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 2" diffuse=".8 .8 .8"/>
    <geom name="table" type="box" pos="0 0 -.025" size=".55 .55 .025" rgba=".12 .16 .22 1"/>
    <geom name="wall_e" type="box" pos=".53 0 .04" size=".01 .55 .04" rgba=".45 .5 .55 1"/>
    <geom name="wall_w" type="box" pos="-.53 0 .04" size=".01 .55 .04" rgba=".45 .5 .55 1"/>
    <geom name="wall_n" type="box" pos="0 .53 .04" size=".55 .01 .04" rgba=".45 .5 .55 1"/>
    <geom name="wall_s" type="box" pos="0 -.53 .04" size=".55 .01 .04" rgba=".45 .5 .55 1"/>
    <site name="station1" type="cylinder" pos="0 0 .003" size=".035 .002" rgba=".2 .8 .4 .7"/>
    <site name="station2" type="cylinder" pos="0 0 .004" size=".035 .002" rgba="1 .65 .1 .7"/>
    <body name="puck" pos="0 0 .035">
      <joint name="x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="y" type="slide" axis="0 1 0" damping="0"/>
      <geom name="puck_geom" type="cylinder" size=".025 .02" mass="1" rgba=".25 .65 1 1" friction="0 0 0"/>
      <site name="center" pos="0 0 .022" size=".006" rgba="1 1 1 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="force_x" joint="x" gear=".4" ctrllimited="true" ctrlrange="-4 4"/>
    <motor name="force_y" joint="y" gear=".4" ctrllimited="true" ctrlrange="-4 4"/>
  </actuator>
</mujoco>'''


@dataclass(frozen=True)
class Layout:
    start: tuple[int, int]
    first: tuple[int, int]
    second: tuple[int, int]

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def datasets():
    rng = random.Random(20261025)
    points = list(itertools.product(range(-2, 3), repeat=2))
    layouts, seen = [], set()
    while len(layouts) < 704:
        layout = Layout(*rng.sample(points, 3))
        if layout.key not in seen:
            seen.add(layout.key)
            layouts.append(layout)
    return {'train': layouts[:512], 'validation': layouts[512:576], 'test': layouts[576:]}


def meters(point):
    return tuple(round(x*.1, 1) for x in point)


def prompt(layout):
    return f'''Control a 1 kg puck in a MuJoCo simulation. Dock at station 1, then station 2.
Initial position: {meters(layout.start)} meters. Initial velocity: (0, 0) m/s.
Station 1: {meters(layout.first)} meters.
Station 2: {meters(layout.second)} meters.
API:
thrust(fx, fy): apply force (0.4*fx, 0.4*fy) newtons for 0.5 seconds. fx and fy must be literal integers from -4 to 4. Both axes act simultaneously.
dock(): check that the puck is within 0.015 m of the next station and its speed is below 0.025 m/s; then coast for 0.25 seconds. Call dock once at each station, in order. A failed dock ends the program.
Physics: there is NO friction or automatic braking. Velocity persists between calls. A thrust changes position by 0.5*current_velocity + 0.05*(fx, fy) meters, and velocity by 0.2*(fx, fy) m/s.
Calibration: starting at rest, two equal and opposite thrust pulses move the puck by 0.1 times the FIRST pulse's (fx, fy) meters and return it to rest.
The puck must stay inside x,y = [-0.48, 0.48] meters throughout. Finish stopped at station 2; a final 0.5 second coast is checked too.
Use exactly SIX calls in this order: thrust, thrust, dock, thrust, thrust, dock.
For EACH leg: accelerate toward that station with one thrust, BRAKE with the exact opposite thrust, then dock. Compute each leg's displacement from its own starting station, not from the original start.
Output only those six API calls with literal integer arguments. No imports, variables, loops, functions, prints, or other calls.'''


def parse_program(text):
    if len(text) > 4096:
        raise ValueError('program_too_long')
    source = text.strip()
    if source.startswith('```'):
        lines = source.splitlines()
        if lines[0] not in ('```', '```python') or lines[-1] != '```':
            raise ValueError('invalid_fence')
        source = '\n'.join(lines[1:-1])
    tree = ast.parse(source)
    if not 1 <= len(tree.body) <= 10:
        raise ValueError('call_budget')
    calls = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ValueError('only_api_calls')
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.func.id not in ('thrust', 'dock') or call.keywords:
            raise ValueError('invalid_api')
        if len(call.args) != (2 if call.func.id == 'thrust' else 0):
            raise ValueError('argument_count')
        args = []
        for node in call.args:
            sign = 1
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                sign, node = -1, node.operand
            if not isinstance(node, ast.Constant) or type(node.value) is not int:
                raise ValueError('literal_integers_only')
            value = sign*node.value
            if abs(value) > 4:
                raise ValueError('force_range')
            args.append(value)
        calls.append((call.func.id, args))
    return calls


@lru_cache(maxsize=1)
def model():
    return mujoco.MjModel.from_xml_string(XML)


def execute(layout, text, record_frames=False):
    m = model()
    d = mujoco.MjData(m)  # fresh simulator state for every generated program
    d.qpos[:] = meters(layout.start)
    mujoco.mj_forward(m, d)
    targets = [np.array(meters(layout.first)), np.array(meters(layout.second))]
    origins = [np.array(meters(layout.start)), targets[0]]
    initial_distances = [np.linalg.norm(t-o) for t, o in zip(targets, origins)]
    best_progress, best_stopped = [0., 0.], [0., 0.]
    docked, trace, frames, error = 0, [], [], None
    call_index = -1

    def snapshot():
        return {'time': float(d.time), 'position': d.qpos.tolist(), 'velocity': d.qvel.tolist(),
                'docked': docked, 'call_index': call_index}

    def measure():
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or any(w.number for w in d.warning):
            raise ValueError('physics_failure')
        if np.max(np.abs(d.qpos)) > .48:
            raise ValueError('out_of_bounds')
        if docked < 2:
            distance = float(np.linalg.norm(d.qpos-targets[docked]))
            progress = max(0., 1.-distance/initial_distances[docked])
            stopped = max(0., 1.-distance/.05)*max(0., 1.-np.linalg.norm(d.qvel)/.1)
            best_progress[docked] = max(best_progress[docked], progress)
            best_stopped[docked] = max(best_stopped[docked], float(stopped))

    def step(count, controls):
        d.ctrl[:] = controls
        for tick in range(count):
            mujoco.mj_step(m, d)
            if record_frames and tick % 5 == 4:
                frames.append(snapshot())
            measure()

    if record_frames:
        frames.append(snapshot())
    try:
        calls = parse_program(text)
        for call_index, (name, args) in enumerate(calls):
            if name == 'thrust':
                step(PULSE_STEPS, args)
            else:
                if docked == 2:
                    raise ValueError('extra_dock')
                if np.linalg.norm(d.qpos-targets[docked]) > POSITION_TOLERANCE:
                    raise ValueError('dock_position')
                if np.linalg.norm(d.qvel) > SPEED_TOLERANCE:
                    raise ValueError('dock_speed')
                # A dock does not snap, clamp, or zero the puck's velocity.
                step(DOCK_STEPS, [0, 0])
                if np.linalg.norm(d.qpos-targets[docked]) > POSITION_TOLERANCE:
                    raise ValueError('dock_drift')
                docked += 1
            trace.append({'call': name, 'args': args, **snapshot()})
        call_index = len(calls)
        step(PULSE_STEPS, [0, 0])
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        error = str(exc)
    final_distance = float(np.linalg.norm(d.qpos-targets[1]))
    speed = float(np.linalg.norm(d.qvel))
    success = bool(error is None and docked == 2 and final_distance <= POSITION_TOLERANCE and speed <= SPEED_TOLERANCE)
    reward = .1*sum(best_progress)+.05*sum(best_stopped)+.3*docked+.1*success
    result = {'reward': float(reward), 'success': success, 'valid': error is None,
              'error': error, 'trace': trace, 'final_state': snapshot(),
              'final_distance_m': final_distance, 'final_speed_m_s': speed,
              'components': {'progress': best_progress, 'stopped': best_stopped, 'docked': docked}}
    if record_frames:
        frames.append(snapshot())
        result['frames'] = frames
    return result


def oracle(layout):
    """Feasibility/unit testing only; never used by the trainer or prompts."""
    current, calls = layout.start, []
    for target in (layout.first, layout.second):
        dx, dy = (t-c for t, c in zip(target, current))
        calls += [f'thrust({dx}, {dy})', f'thrust({-dx}, {-dy})', 'dock()']
        current = target
    return '\n'.join(calls)
