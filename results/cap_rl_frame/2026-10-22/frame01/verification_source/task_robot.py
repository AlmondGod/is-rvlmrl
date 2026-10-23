"""CaP-RL frame subtask: center-grasp and lift a free wooden rail with an xArm7."""
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import random

import numpy as np
from .robot_lift import Robot, ORIGIN, HEIGHT

SYSTEM = 'Write only the requested Python API program. Do not explain it.'


@dataclass(frozen=True)
class Layout:
    start: tuple[int, int]
    center: tuple[int, int]
    lift_cm: int

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def datasets():
    rng = random.Random(20261026)
    layouts, seen = [], set()
    while len(layouts) < 448:
        layout = Layout(tuple(rng.randint(-3, 3) for _ in range(2)),
                        tuple(rng.randint(-4, 4) for _ in range(2)), rng.choice([6, 8, 10, 12]))
        if layout.key not in seen and layout.start != layout.center:
            layouts.append(layout); seen.add(layout.key)
    return {'train': layouts[:256], 'validation': layouts[256:320], 'test': layouts[320:]}


def prompt(layout):
    return f'''Use the xArm7 robot to grasp the CENTER of one wooden frame rail and lift it {layout.lift_cm} cm above its starting height. Keep holding it steady.
The rail is 91.44 cm long, 3.8 cm wide, and 6.4 cm tall. Gravity and finger contact act in MuJoCo.
All XY coordinates below are centimeters relative to the workbench reference.
Gripper starts above XY {layout.start}, open, at a safe height.
Rail CENTER is at XY {layout.center}.
API:
move(dx, dy): move gripper sideways by RELATIVE displacement in cm. Literal integer arguments from -7 to 7. It does not descend or grasp.
lower(): lower at the current XY to the calibrated grasp height. It does not move sideways or close fingers.
close(): close fingers and wait for physical contact. It does not move or lift the rail.
lift(h): move upward by h cm, a literal integer from 1 to 14, and hold.
Use FOUR calls in order: move, lower, close, lift. For move, subtract the starting gripper XY from the rail CENTER XY. For lift, use the requested lift amount.
Success requires a central grasp, the requested physical rail height within 8 mm, a level rail, low speed, and sustained finger contact during the final half-second. No object snapping or attachment is provided.
Output only the four API calls, with literal integer arguments. No variables, imports, loops, functions, prints, or explanations.'''


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
    if not 1 <= len(tree.body) <= 6:
        raise ValueError('call_budget')
    calls = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            raise ValueError('only_api_calls')
        c = stmt.value
        if not isinstance(c.func, ast.Name) or c.func.id not in ('move', 'lower', 'close', 'lift') or c.keywords:
            raise ValueError('invalid_api')
        n = {'move': 2, 'lower': 0, 'close': 0, 'lift': 1}[c.func.id]
        if len(c.args) != n:
            raise ValueError('argument_count')
        args = []
        for node in c.args:
            sign = 1
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                node, sign = node.operand, -1
            if not isinstance(node, ast.Constant) or type(node.value) is not int:
                raise ValueError('literal_integers_only')
            v = sign*node.value
            if (c.func.id == 'move' and abs(v) > 7) or (c.func.id == 'lift' and not 1 <= v <= 14):
                raise ValueError('argument_range')
            args.append(v)
        calls.append((c.func.id, args))
    return calls


def execute(layout, text, record_frames=False):
    result = {'success': False, 'reward': 0., 'valid': False, 'error': None, 'trace': []}
    try:
        calls = parse_program(text)
    except (ValueError, SyntaxError) as e:
        result['error'] = str(e)
        return result
    r = Robot(*layout.center, start=layout.start, record=record_frames)
    desired = HEIGHT/2 + layout.lift_cm*.01
    alignment = contact = elevation = 0.

    def update():
        nonlocal alignment, contact, elevation
        s = r.state()
        err = float(np.linalg.norm(np.array(s['tcp'][:2])-s['wood'][:2]))
        alignment = max(alignment, max(0., 1-err/.06))
        central = err <= .012
        contact = max(contact, float(s['grip_force'] > 5 and central))
        elevation = max(elevation, float(np.clip((s['wood'][2]-HEIGHT/2)/(layout.lift_cm*.01), 0, 1))*float(central and s['grip_force'] > 5))
        return s, central

    error = None
    for i, (name, args) in enumerate(calls):
        r.call_index = i
        try:
            getattr(r, 'shift' if name == 'move' else name)(*args)
        except ValueError as e:
            error = str(e)
        state, _ = update()
        result['trace'].append({'call': name, 'args': args, 'time': float(r.d.time), **state})
        if error:
            break
    stable = True
    if not error:
        for _ in range(25):
            r.tick(r.target.copy())
            s, central = update()
            stable &= (central and abs(s['wood'][2]-desired) <= .008 and s['tilt'] <= .10
                       and np.linalg.norm(s['wood_velocity'][:3]) <= .02
                       and np.linalg.norm(s['wood_velocity'][3:]) <= .10 and s['grip_force'] > 5)
    safe = r.peak < .003 and all(w.number == 0 for w in r.d.warning)
    success = bool(not error and safe and stable)
    components = {'alignment': .15*alignment, 'contact': .20*contact,
                  'elevation': .15*elevation, 'success': .50*success}
    result.update(success=success, reward=sum(components.values()) if safe else 0., valid=error is None,
                  error=error or (None if safe else 'penetration_limit'), final=r.state(),
                  components=components, time=float(r.d.time))
    if record_frames:
        r.sample()
        result['frames'] = r.frames
    return result


def oracle(layout):
    return f'move({layout.center[0]-layout.start[0]}, {layout.center[1]-layout.start[1]})\nlower()\nclose()\nlift({layout.lift_cm})'
