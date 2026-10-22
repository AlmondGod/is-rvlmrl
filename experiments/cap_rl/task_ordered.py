"""Two-object ordered delivery: longer programs and cross-delivery state tracking.

A symbolic integer grid, not physics. The task has fresh deterministic splits.
The original one-object task and its reward remain byte-for-byte unchanged.
"""
import ast
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import random

from .task import SYSTEM, parse_program as parse_short_program


@dataclass(frozen=True)
class Layout:
    gripper: tuple[int, int]
    red: tuple[int, int]
    blue: tuple[int, int]
    red_target: tuple[int, int]
    blue_target: tuple[int, int]
    first: str

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def datasets():
    rng = random.Random(20261024)
    points = list(itertools.product(range(-2, 3), repeat=2))
    layouts, seen = [], set()
    while len(layouts) < 704:
        layout = Layout(*rng.sample(points, 5), first=rng.choice(('red', 'blue')))
        if layout.key not in seen:
            seen.add(layout.key)
            layouts.append(layout)
    return {'train': layouts[:512], 'validation': layouts[512:576], 'test': layouts[576:]}


def prompt(layout):
    second = 'blue' if layout.first == 'red' else 'red'
    return f"""Deliver BOTH objects to their matching targets and release them, in the required order.
Gripper starts at {layout.gripper}.
Delivery 1 (FIRST): {layout.first} object is at {getattr(layout, layout.first)}; its target is {getattr(layout, layout.first+'_target')}.
Delivery 2 (SECOND): {second} object is at {getattr(layout, second)}; its target is {getattr(layout, second+'_target')}.
Finish delivery 1, including release at its target, before picking up delivery 2.
The board coordinates are integers from -2 to 2 on both axes.
Available API:
move(dx, dy): move the gripper by RELATIVE integer offsets dx and dy; a held object moves with it. Both axes may change in one call. Stay on the board.
pick(): grasp the object at the gripper's current coordinates; the gripper must be empty. Picking the second object before the first is correctly released is invalid. Do not pick up a delivered object.
place(): release the held object at the gripper's current coordinates. Do not place on another object.
After place(), the gripper stays at that position: all subsequent moves are relative to its CURRENT position.
Write one complete program of at most 12 calls using move, pick, place with literal integers only.
No imports, variables, functions, loops, prints, or other API calls.
Both objects must finish at their own targets and the gripper must finish empty."""


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
    if not 1 <= len(tree.body) <= 12:
        raise ValueError('call_budget')
    # Reuse exactly the original safe expression grammar, with a larger budget.
    return [parse_short_program(ast.unparse(statement))[0] for statement in tree.body]


def distance(a, b):
    return sum(abs(x-y) for x, y in zip(a, b))


def execute(layout, text):
    gripper = layout.gripper
    objects = {'red': layout.red, 'blue': layout.blue}
    targets = {'red': layout.red_target, 'blue': layout.blue_target}
    order = [layout.first, 'blue' if layout.first == 'red' else 'red']
    approach_initial = {order[0]: distance(gripper, objects[order[0]]),
                        order[1]: distance(targets[order[0]], objects[order[1]])}
    best_approach = dict(approach_initial)
    picked, delivered = set(), set()
    holding, trace, error = None, [], None
    try:
        for name, args in parse_program(text):
            active = next((color for color in order if color not in delivered), None)
            if name == 'move':
                destination = tuple(x+d for x, d in zip(gripper, args))
                if any(abs(x) > 2 for x in destination):
                    raise ValueError('out_of_bounds')
                gripper = destination
                if holding:
                    objects[holding] = gripper
                elif active and active not in picked:
                    best_approach[active] = min(best_approach[active], distance(gripper, objects[active]))
            elif name == 'pick':
                color = next((c for c, position in objects.items() if position == gripper), None)
                if holding or color is None:
                    raise ValueError('invalid_pick')
                if color in delivered:
                    raise ValueError('already_delivered')
                if color != active:
                    raise ValueError('wrong_order')
                holding = color
                picked.add(color)
            else:
                if holding is None:
                    raise ValueError('empty_place')
                if any(c != holding and position == gripper for c, position in objects.items()):
                    raise ValueError('occupied_place')
                if gripper == targets[holding]:
                    delivered.add(holding)
                holding = None
            trace.append({'call': name, 'args': args, 'gripper': gripper,
                          'objects': dict(objects), 'holding': holding,
                          'delivered': sorted(delivered)})
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        error = str(exc)
    success = error is None and len(delivered) == 2 and holding is None and objects == targets
    components = {}
    for color in order:
        approach = 1. - best_approach[color] / approach_initial[color]
        progress = max(0., 1. - distance(objects[color], targets[color]) /
                       distance(getattr(layout, color), targets[color]))
        components[color] = {'approach': approach, 'picked': color in picked,
                             'transport': progress, 'delivered': color in delivered}
    reward = sum(.075*c['approach'] + .1*c['picked'] + .075*c['transport'] +
                 .2*c['delivered'] for c in components.values()) + .1*success
    # Preserve prefix credit after runtime errors: discounting it rewards stopping
    # after the first delivery rather than attempting the second. Runtime errors
    # still stop execution, invalidate the program, and forfeit the success bonus.
    # Syntax errors have no executed prefix; truncation is zeroed by the trainer.
    return {'reward': reward, 'success': success, 'valid': error is None,
            'error': error, 'trace': trace}


def oracle(layout):
    """Feasibility tests only; never used by training or included in prompts."""
    order = [layout.first, 'blue' if layout.first == 'red' else 'red']
    gripper, calls = layout.gripper, []
    for color in order:
        position, target = getattr(layout, color), getattr(layout, color+'_target')
        calls += [f'move({position[0]-gripper[0]}, {position[1]-gripper[1]})', 'pick()',
                  f'move({target[0]-position[0]}, {target[1]-position[1]})', 'place()']
        gripper = target
    return '\n'.join(calls)
