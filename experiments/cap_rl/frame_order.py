"""Additional pickup-first verification, fixed before inspecting held-out results.

The frozen frame01 reward permits slide-then-lift. Keep that training source and
its recorded rewards intact, but do not count such programs as ordered carry.
This check uses actual simulator call-boundary states, independently replayed by
the evidence audits. It does not inspect canonical code or model weights.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random

from .task_frame import datasets, ORIGIN, HEIGHT


def ordered_carry(layout, trace):
    source = [float(o)+v*.01 for o,v in zip(ORIGIN, layout.center)]
    target = [float(o)+v*.01 for o,v in zip(ORIGIN, layout.target)]
    pickup_index = arrival_index = None
    for i, state in enumerate(trace):
        central = math.dist(state['tcp'][:2], state['wood'][:2]) <= .012
        raised = state['wood'][2]-HEIGHT/2 >= layout.lift_cm*.01-.008
        grasped = central and state['grip_force'] > 5
        if not (raised and grasped):
            continue
        if pickup_index is None and math.dist(state['wood'][:2], source) <= .012:
            pickup_index = i
        if pickup_index is not None and i > pickup_index and math.dist(state['wood'][:2], target) <= .008:
            arrival_index = i
            break
    return {'passed': arrival_index is not None, 'pickup_call_index': pickup_index,
            'arrival_call_index': arrival_index}


def verify_report(directory):
    directory = Path(directory)
    original = json.loads((directory/'test_comparison.json').read_text())
    report = copy.deepcopy(original)
    layouts = datasets()['test']
    excluded = {}
    for label, phase in [('baseline','before'),('trained','after')]:
        rows = report[phase]
        if [r['layout_id'] for r in rows] != [l.key for l in layouts]:
            raise ValueError('Test split/order mismatch')
        removed = []
        for i,(layout,row) in enumerate(zip(layouts,rows)):
            check = ordered_carry(layout,row['trace'])
            row['original_success'] = row['success']
            row['ordered_carry'] = check
            row['success'] = bool(row['success'] and check['passed'])
            if row['original_success'] and not row['success']:
                removed.append(i+1)
        # Reward is deliberately preserved: this is an extra success criterion,
        # not a retrospective rewrite of the reward the optimizer received.
        report[label] = {**original[label], 'successes':sum(r['success'] for r in rows),
                         'success_rate':sum(r['success'] for r in rows)/len(rows)}
        excluded[phase] = removed
    pairs=list(zip(report['before'],report['after']))
    deltas=[int(b['success'])-int(a['success']) for a,b in pairs]
    rng=random.Random(20261023)
    boot=sorted(sum(rng.choices(deltas,k=len(deltas)))/len(deltas) for _ in range(10000))
    report.update(paired_wins=sum(not a['success'] and b['success'] for a,b in pairs),
                  paired_losses=sum(a['success'] and not b['success'] for a,b in pairs),
                  delta_success_rate=sum(deltas)/len(deltas),
                  paired_bootstrap_95_ci=[boot[250],boot[9749]],
                  verification={'criterion':'Raised and centrally grasped within 12 mm of original rail center; a later call boundary must be raised and grasped within 8 mm of target. Original final-placement criteria must also pass.',
                                'defined':'After training, before inspecting held-out results',
                                'excluded_original_success_indices_one_based':excluded,
                                'original_training_rewards_preserved':True})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    report=verify_report(args.directory)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('before','after')},indent=2))


if __name__=='__main__':
    main()
