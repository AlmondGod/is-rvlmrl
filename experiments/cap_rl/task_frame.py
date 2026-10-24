"""Physical frame-assembly subset: pick, transfer, place and release one rail."""
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import random

import numpy as np
from .robot_lift import Robot as LiftRobot, ORIGIN, HEIGHT

SYSTEM = 'Write only the requested Python API program. Do not explain it.'


class Robot(LiftRobot):
    def open(self):
        self.closed = 0.
        self.hold(2.)

    def retreat(self):
        goal = self.target.copy()
        goal[2] = .20
        self.move(goal)
        self.hold(1.)

    def state(self):
        state = super().state()
        rotation = self.d.xmat[self.bid].reshape(3, 3)
        state['yaw'] = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        state['gripper_command'] = float(self.closed)
        return state


@dataclass(frozen=True)
class Layout:
    start: tuple[int, int]
    center: tuple[int, int]
    target: tuple[int, int]
    lift_cm: int

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def datasets():
    rng = random.Random(20261027)
    layouts, seen = [], set()
    while len(layouts) < 704:
        l = Layout(tuple(rng.randint(-3, 3) for _ in range(2)),
                   tuple(rng.randint(-4, 4) for _ in range(2)),
                   tuple(rng.randint(-4, 4) for _ in range(2)), rng.choice([6, 8, 10]))
        distance = np.linalg.norm(np.array(l.target)-l.center)
        if l.key not in seen and l.start != l.center and 3 <= distance <= 7:
            layouts.append(l); seen.add(l.key)
    return {'train': layouts[:512], 'validation': layouts[512:576], 'test': layouts[576:]}


def prompt(l):
    return f'''Place one wooden rail at its frame assembly target using the xArm7 robot.
The free rail is 91.44 cm long, 3.8 cm wide, 6.4 cm tall. Gravity and physical finger contact act in MuJoCo; there is no object attachment.
All XY coordinates are centimeters relative to the workbench reference.
Open gripper starts above XY {l.start}. Rail CENTER starts at XY {l.center}.
Target rail CENTER is XY {l.target}. Keep its initial orientation. Required lift clearance is {l.lift_cm} cm.
API (only literal integer arguments):
move(dx, dy): shift gripper by RELATIVE XY displacement in cm, each from -8 to 8; keeps current height.
lower(): descend at current XY to the calibrated grasp/placement height; does not open fingers.
close(): close fingers and wait for contact.
lift(h): rise by h cm, from 1 to 14, and hold.
open(): open fingers and wait; does not move the arm.
retreat(): withdraw vertically to safe height and wait.
Plan the pickup, lift, transfer, lowering, release, and retreat. The rail must reach the requested clearance while grasped before arriving above its target. Then it must rest on the bench within 8 mm of the target, level, correctly oriented, stationary, with fingers open, no finger contact, and the gripper withdrawn.
Output only API calls. No variables, calculations in arguments, imports, loops, functions, prints, or explanations. Maximum ten calls.'''


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
    arity = {'move': 2, 'lower': 0, 'close': 0, 'lift': 1, 'open': 0, 'retreat': 0}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            raise ValueError('only_api_calls')
        call = stmt.value
        if not isinstance(call.func, ast.Name) or call.func.id not in arity or call.keywords:
            raise ValueError('invalid_api')
        name = call.func.id
        if len(call.args) != arity[name]:
            raise ValueError('argument_count')
        args = []
        for node in call.args:
            sign = 1
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                node, sign = node.operand, -1
            if not isinstance(node, ast.Constant) or type(node.value) is not int:
                raise ValueError('literal_integers_only')
            value = sign*node.value
            if (name == 'move' and abs(value) > 8) or (name == 'lift' and not 1 <= value <= 14):
                raise ValueError('argument_range')
            args.append(value)
        calls.append((name, args))
    return calls


def execute(l, text, record_frames=False):
    result = {'success': False, 'reward': 0., 'valid': False, 'error': None, 'trace': []}
    try:
        calls = parse_program(text)
    except (ValueError, SyntaxError) as exc:
        result['error'] = str(exc)
        return result
    r = Robot(*l.center, start=l.start, record=record_frames)
    target = ORIGIN + np.array(l.target)*.01
    alignment = contact = elevation = transport = placement = 0.
    lifted = arrived = False

    def update():
        nonlocal alignment, contact, elevation, transport, placement, lifted, arrived
        s = r.state()
        wood = np.array(s['wood'])
        central_error = float(np.linalg.norm(np.array(s['tcp'][:2])-wood[:2]))
        distance = float(np.linalg.norm(wood[:2]-target))
        alignment = max(alignment, max(0., 1-central_error/.06))
        grasped = central_error <= .012 and s['grip_force'] > 5
        contact = max(contact, float(grasped))
        height = wood[2]-HEIGHT/2
        elevation = max(elevation, float(np.clip(height/(l.lift_cm*.01), 0, 1))*grasped)
        lifted |= bool(grasped and height >= l.lift_cm*.01-.008)
        if lifted and grasped and height >= l.lift_cm*.01-.008:
            transport = max(transport, max(0., 1-distance/.08))
            arrived |= distance <= .008
        grounded = abs(wood[2]-HEIGHT/2) <= .003
        oriented = s['tilt'] <= .05 and abs(s['yaw']) <= .05
        released = s['gripper_command'] == 0 and s['grip_force'] < 1.
        if arrived and grounded and oriented and released:
            placement = max(placement, max(0., 1-distance/.04))
        settled = (arrived and distance <= .008 and grounded and oriented and released
                   and s['tcp'][2]-wood[2] >= .10
                   and np.linalg.norm(s['wood_velocity'][:3]) <= .01
                   and np.linalg.norm(s['wood_velocity'][3:]) <= .05)
        return s, bool(settled)

    error = None
    for i, (name, args) in enumerate(calls):
        r.call_index = i
        try:
            getattr(r, 'shift' if name == 'move' else name)(*args)
        except ValueError as exc:
            error = str(exc)
        state, _ = update()
        result['trace'].append({'call': name, 'args': args, 'time': float(r.d.time), **state})
        if error:
            break
    stable = True
    if not error:
        for _ in range(25):
            try:
                r.tick(r.target.copy())
            except ValueError as exc:
                error = str(exc)
                break
            _, settled = update()
            stable &= settled
    safe = r.peak < .003 and all(w.number == 0 for w in r.d.warning)
    success = bool(not error and safe and stable)
    components = {'alignment': .10*alignment, 'contact': .15*contact,
                  'elevation': .15*elevation, 'transport': .15*transport,
                  'placement': .10*placement, 'success': .35*success}
    result.update(success=success, reward=sum(components.values()) if safe else 0., valid=error is None,
                  error=error or (None if safe else 'penetration_limit'), final=r.state(),
                  components=components, lifted=lifted, arrived=arrived, time=float(r.d.time))
    if record_frames:
        r.sample()
        result['frames'] = r.frames
    return result


def oracle(l):
    return (f'move({l.center[0]-l.start[0]}, {l.center[1]-l.start[1]})\nlower()\nclose()\n'
            f'lift({l.lift_cm})\nmove({l.target[0]-l.center[0]}, {l.target[1]-l.center[1]})\n'
            'lower()\nopen()\nretreat()')
