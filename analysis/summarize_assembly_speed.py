"""Summarize verified full-episode equivalence and isolated prefix timings."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    root = Path(root)
    status = json.loads((root/'speed-status.json').read_text())
    if status['stage'] != 'complete':
        raise ValueError('Speed experiment is not complete')
    full = json.loads((root/'full-comparison.json').read_text())
    if not (full['exact'] and full['complete_assembly_sequence']):
        raise ValueError('Both full runs must complete with exact recorded-state/result equivalence')
    successful = json.loads((root/'success-comparison.json').read_text())
    if not (successful['exact'] and successful['complete_assembly_sequence']
            and successful['baseline_success'] and successful['optimized_success']):
        raise ValueError('The integrated backend must preserve the known strict success exactly')
    successful = dict(successful)
    successful.pop('speedup')
    successful['timing_caveat'] = ('Historical reference and current integrated run use different hosts/runtime. '
                                  'This is a correctness regression, not a controlled timing comparison.')
    rows = status['isolated_trials']
    before = [r['wall_s'] for r in rows if r['backend'] == 'reference']
    after = [r['wall_s'] for r in rows if r['backend'] == 'optimized']
    if len(before) != 3 or len(after) != 3:
        raise ValueError('Expected three isolated timings for each backend')
    for i in range(3):
        pair = json.loads((root/f'isolated-comparison-{i}.json').read_text())
        if not pair['exact']:
            raise ValueError('Isolated prefix trajectories differ')
    profiles = {}
    for name in ('reference', 'optimized'):
        profile = json.loads((root/f'{name}-41002.profile.json').read_text())
        if not profile['complete']:
            raise ValueError('Incomplete profiler record')
        t = {k: v['seconds'] for k, v in profile['timings'].items()}
        disjoint = dict(physics_kernel=t['physics_kernel'], nail_forces=t['nail_forces'],
                        inverse_kinematics=t['inverse_kinematics'],
                        other_environment_step=t['environment_step']-t['physics_kernel']-t['nail_forces']-t['inverse_kinematics'],
                        camera_render=t['camera_render'], frame_vision=t['frame_vision'],
                        panel_vision=t['panel_vision'], metrics=t['metrics'],
                        controller=t['controller'], acceptance=t['acceptance'])
        disjoint['setup_io_and_other'] = profile['elapsed_s']-sum(disjoint.values())
        if min(disjoint.values()) < -1e-5:
            raise ValueError('Overlapping profiler accounting')
        profiles[name] = dict(exclusive_wall_seconds=disjoint,
                              profiler_total_seconds=profile['elapsed_s'],
                              physics_kernel_fraction=t['physics_kernel']/profile['elapsed_s'],
                              physics_steps=profile['timings']['physics_kernel']['calls'],
                              camera_images=3*profile['timings']['camera_render']['calls'])
    return dict(full_episode=full, successful_regression=successful,
                isolated_prefix=dict(simulated_seconds=20, repetitions_per_backend=3,
                reference_seconds=before, optimized_seconds=after,
                reference_median_seconds=statistics.median(before), optimized_median_seconds=statistics.median(after),
                median_time_ratio=statistics.median(before)/statistics.median(after),
                exact_all_recorded_states_and_results=True,
                scope='Isolated alternating runs of the early 20-second assembly prefix; not whole-episode latency.'),
                profiles=profiles, hardware=json.loads((root/'hardware.json').read_text()),
                scope='Execution acceleration of one fixed model-generated program on training layout 41002, plus a strict-success regression on training layout 41001. No new RL training or held-out policy-improvement claim.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    result = summarize(a.evidence)
    Path(a.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
