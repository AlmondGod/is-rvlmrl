"""The entire model-facing interface. No simulator imports or file access."""
import json

SKILLS = {
    "position_member": "Use the shared RGB/encoder skill to pick, transport and align member index 0..3; does not release.",
    "release_member": "Release the positioned member index 0..3 and retreat. Must have been positioned.",
    "park_grippers": "After all four members are released, move both gripper arms to the verified home clearances.",
    "nail_corner": "Visit corner index 0 SW, 1 SE, 2 NE, 3 NW once. Requires four released members and parked grippers.",
    "feed_panel": "Move the physical plywood feeder into the workspace after four corner nail triggers.",
    "position_panel": "Prepare both grippers, grasp, lift, retract shelves, align and lower the sheet; does not release.",
    "release_panel": "Release the positioned sheet and retreat through planned joint paths.",
    "nail_panel": "Visit sheet-to-member index 0..3 once after sheet release.",
    "flip_structure": "Regrasp the stack, lift clear, rotate 180 degrees and hold inverted for five seconds. Requires all eight triggers.",
    "finish": "Declare the full assembly and flip visually complete.",
    "stop": "End the trial if continuation is inappropriate."
}
INDEXED = {"position_member", "release_member", "nail_corner", "nail_panel"}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "skill": {"type": "string", "enum": list(SKILLS)},
        "index": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 3}, {"type": "null"}]},
        "reason": {"type": "string"}
    },
    "required": ["skill", "index", "reason"]
}
SYSTEM = """You control a simulated construction robot through the supplied skills.
Use the current camera images and sensor packet to choose exactly ONE next skill.
Return only a JSON object with keys skill, index, reason. No markdown, other tools,
file access or multiple-action sequences. The reason must describe visible evidence
and why the action is appropriate. Use null index for unindexed skills.
The task is four loose lumber members -> flat frame -> four corner nail triggers ->
feed, place and release plywood -> four panel nail triggers -> flip -> inspect -> finish.
Inspect fresh images after each completed skill. Skill completion is not proof of
correct assembly. Nail triggers do not report whether the hidden joint is sound.
Member identities are roles from shared visual discovery, not fixed feeder lanes:
0 long left-side member to front, 1 short left-side member to left,
2 long right-side member to rear, 3 short right-side member to right.
The supplied skills implement perception, grasping, planning and fast feedback.
You choose the next action, inspect results and decide whether to stop.
Known geometry and camera calibration are permitted; true object poses and joint
condition are unavailable. No colored markers. Stop if a failed skill prevents
continuation. This is an uncalibrated simulation, not a physical robot.
"""

def validate_action(text):
    """Strict parse; do not repair or silently extract a favorable candidate."""
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"skill", "index", "reason"}:
        raise ValueError("action_schema")
    if value["skill"] not in SKILLS:
        raise ValueError("unknown_skill")
    index = value["index"]
    if value["skill"] in INDEXED:
        if type(index) is not int or not 0 <= index < 4:
            raise ValueError("index_range")
    elif index is not None:
        raise ValueError("unexpected_index")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("empty_reason")
    return value

def public_robot(robot):
    return {key: robot[key] for key in (
        "time", "tcp", "rotation", "joints", "joint_velocity",
        "gripper_command", "shuttle_x", "rack_slide0", "rack_slide1"
    )}

def public_events(events):
    return [{k: v for k, v in event.items()
             if k in ("kind", "index", "time", "tcp", "triggered")} for event in events]

def response_ready(observation_sim_s, latency_wall_s, current_sim_s):
    return current_sim_s + 1e-9 >= observation_sim_s + latency_wall_s
