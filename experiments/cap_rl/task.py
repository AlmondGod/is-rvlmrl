"""Deterministic discrete pick/place task; deliberately NOT a physics simulator.

Only a tiny Python call grammar is interpreted. Generated text never reaches exec.
The model observes integer coordinates, not camera images. No assembly-benchmark
acceptance rules or historical results are changed by this separate experiment.
"""
import ast
import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Layout:
    gripper: tuple[int, int]
    member: tuple[int, int]
    target: tuple[int, int]

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def datasets():
    """Fixed, disjoint layouts, frozen before any model evaluation."""
    points = list(itertools.product(range(-2, 3), repeat=2))
    layouts = [Layout(*triple) for triple in itertools.permutations(points, 3)]
    random.Random(20261022).shuffle(layouts)
    return {"train": layouts[:256], "validation": layouts[256:320],
            "test": layouts[320:448]}


SYSTEM = "You write short robot control programs. Return only Python calls, without explanations."


def prompt(layout):
    return f"""Move the wooden member to the target and release it.
Gripper starts at {layout.gripper}. Member is at {layout.member}. Target is {layout.target}.
The board coordinates are integers from -2 to 2 on both axes.
Available API:
move(dx, dy): move the gripper by RELATIVE integer offsets dx and dy; a held member moves with it. Both axes may change in one call. Stay on the board.
pick(): grasp the member; the gripper must be at the member's current coordinates and empty.
place(): release the held member at the gripper's current coordinates.
Write at most 8 calls using move, pick, place with literal integers only.
No imports, variables, functions, loops, prints, or other API calls.
The member must finish at the target and the gripper must finish empty."""


def parse_program(text):
    if len(text) > 4096:
        raise ValueError("program_too_long")
    source = text.strip()
    if source.startswith("```"):
        lines = source.splitlines()
        if lines[0] not in ("```", "```python") or lines[-1] != "```":
            raise ValueError("invalid_fence")
        source = "\n".join(lines[1:-1])
    tree = ast.parse(source)
    if not 1 <= len(tree.body) <= 8:
        raise ValueError("call_budget")
    calls = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ValueError("only_calls")
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.func.id not in ("move", "pick", "place"):
            raise ValueError("unknown_api")
        if call.keywords or len(call.args) != (2 if call.func.id == "move" else 0):
            raise ValueError("arguments")
        args = []
        for node in call.args:
            sign = 1
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                sign, node = -1, node.operand
            if not isinstance(node, ast.Constant) or type(node.value) is not int:
                raise ValueError("literal_integers_only")
            value = sign * node.value
            if abs(value) > 4:
                raise ValueError("offset_range")
            args.append(value)
        calls.append((call.func.id, args))
    return calls


def execute(layout, text):
    gripper, member = layout.gripper, layout.member
    holding = picked = False
    trace = []
    approach_distance = sum(abs(a-b) for a,b in zip(gripper, member))
    best_approach = approach_distance
    try:
        calls = parse_program(text)
        for name, args in calls:
            if name == "move":
                gripper = tuple(x + delta for x, delta in zip(gripper, args))
                if any(abs(x) > 2 for x in gripper):
                    raise ValueError("out_of_bounds")
                if holding:
                    member = gripper
                elif not picked:
                    best_approach = min(best_approach, sum(abs(a-b) for a,b in zip(gripper, member)))
            elif name == "pick":
                if holding or gripper != member:
                    raise ValueError("invalid_pick")
                holding = picked = True
            else:
                if not holding:
                    raise ValueError("empty_place")
                holding = False
            trace.append({"call": name, "args": args, "gripper": gripper,
                          "member": member, "holding": holding})
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        return {"reward": 0., "success": False, "valid": False,
                "error": str(exc), "trace": trace}
    success = picked and not holding and member == layout.target
    initial_distance = sum(abs(a-b) for a,b in zip(layout.member, layout.target))
    final_distance = sum(abs(a-b) for a,b in zip(member, layout.target))
    progress = max(0., 1. - final_distance / initial_distance)
    approach = 1. - best_approach / approach_distance
    reward = .15 * approach + .2 * picked + .15 * progress + .5 * success
    return {"reward": reward, "success": success, "valid": True,
            "error": None, "trace": trace}


def oracle(layout):
    """Test-only feasibility oracle, never called by the trainer or prompt."""
    a = [m-g for g,m in zip(layout.gripper, layout.member)]
    b = [t-m for m,t in zip(layout.member, layout.target)]
    return f"move({a[0]}, {a[1]})\npick()\nmove({b[0]}, {b[1]})\nplace()"
