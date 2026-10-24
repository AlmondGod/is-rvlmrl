"""Training-only feasibility probe for a camera-conditioned assembly prompt.

This does not train or change the active policy. The extraction function accepts
only RGB frames, camera calibration and the public robot packet. Scene identity
is retained in the receipt, never included in the proposed policy observation.
"""
import argparse
import hashlib
import json
from pathlib import Path


def extract(frames, cameras, robot):
    import numpy as np
    from experiments.torso_assembly.frame_vision import MultiVision
    estimator = MultiVision()
    estimates = estimator.update(frames, cameras, robot)
    members = []
    for index, estimate in enumerate(estimates):
        if estimate is None:
            members.append(dict(member=index, observed=False))
            continue
        center = np.asarray(estimate['center'])
        yaw = float(estimate['yaw'])
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(yaw):
            raise ValueError('Nonfinite or malformed visual estimate')
        members.append(dict(member=index, observed=True,
            x_m=round(float(center[0]), 4), y_m=round(float(center[1]), 4),
            yaw_deg=round(float(np.degrees(yaw)), 2)))
    return dict(format='initial_rgb_member_estimates_v1', time_s=float(robot['time']),
        source='Shared marker-free RGB estimator, calibrated cameras and known inventory/support priors.',
        members=members)


def describe(observation):
    """Whitelist fields; scene IDs, truth poses and outcomes cannot enter text."""
    if observation['format'] != 'initial_rgb_member_estimates_v1':
        raise ValueError('Unknown observation schema')
    members = observation['members']
    if [m['member'] for m in members] != list(range(4)):
        raise ValueError('Expected four visual member roles in order')
    lines = ['Initial camera estimates in the calibrated workspace frame (metres and degrees).',
             'These are visual estimates, not exact object poses. Missing observations remain unknown.']
    for member in members:
        i = member['member']
        if not member['observed']:
            lines.append(f'Member {i}: not visually localized.')
        else:
            lines.append(f"Member {i}: x={member['x_m']:.4f} m, y={member['y_m']:.4f} m, yaw={member['yaw_deg']:.2f} degrees.")
    for label, indices in (('Left', (0, 1)), ('Right', (2, 3))):
        pair = [members[i] for i in indices]
        if all(m['observed'] for m in pair):
            pair.sort(key=lambda m: abs(m['x_m']))
            if abs(abs(pair[0]['x_m'])-abs(pair[1]['x_m'])) >= .01:
                lines.append(f"{label} side, nearer to farther from workspace centre: members {pair[0]['member']}, {pair[1]['member']}.")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.seed not in range(51001, 51033):
        raise ValueError('This feasibility probe is restricted to the declared training layouts')
    import cv2
    import imageio.v2 as imageio
    import mujoco
    from experiments.arm_corner.run import clean
    from experiments.assembly_benchmark.interface import public_robot
    from experiments.assembly_benchmark.run import robot
    from experiments.cap_rl.assembly import PROTOCOL, provenance
    from experiments.torso_assembly.acceleration import configure
    from experiments.torso_assembly.environment import Env
    cv2.setNumThreads(1)
    configure('vectorized')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(PROTOCOL.read_text())['environment']
    env = Env(dt=config['dt'], layout_seed=args.seed, physical_nails=True,
              fixture_half_width=config['fixture_half_width'],
              shoulder_half_span=config['shoulder_half_span'])
    if env.d.time != 0.: raise ValueError('Expected the untouched initial observation')
    renderer = mujoco.Renderer(env.m, config['image_height'], config['image_width'])
    try:
        frames, cameras = env.images(renderer)
        packet = public_robot(robot(env))
        observation = extract(frames, cameras, packet)
        for i in (0, 2, 3): imageio.imwrite(out/f'camera{i}.png', frames[i])
    finally:
        renderer.close()
    (out/'sensor_inputs.json').write_text(json.dumps(clean(dict(cameras=cameras, robot=packet)), indent=2)+'\n')
    (out/'observation.json').write_text(json.dumps(observation, indent=2)+'\n')
    (out/'prompt_fragment.txt').write_text(describe(observation)+'\n')
    receipt = dict(scope='Observation feasibility only; no policy update, task reward or held-out evaluation.',
        seed=args.seed, observation_time_s=float(env.d.time),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_provenance=provenance(support_profile='contact_and_nails', controller_profile='southwest_approach'),
        files_sha256={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in out.iterdir() if q.is_file()})
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(observation), flush=True)


if __name__ == '__main__': main()
