"""Audit full-assembly record provenance, reward accounting and split isolation.

This verifies saved evidence, not a new physics execution or model generation.
Use audit_assembly for independent physical replay, and generation audit below
for independent greedy checkpoint regeneration.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from experiments.cap_rl import task_assembly as task
from experiments.cap_rl.assembly import parse_program, score, PROTOCOL
from experiments.cap_rl.audit_assembly import compare


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(rows):
    return dict(n=len(rows), successes=sum(r['success'] for r in rows),
        success_rate=sum(r['success'] for r in rows)/len(rows),
        mean_reward=sum(r['reward'] for r in rows)/len(rows),
        valid_rate=sum(r['valid'] for r in rows)/len(rows))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory):
    directory = Path(directory)
    metadata = read(directory/'metadata.json')
    require(metadata['task'] == 'assembly_full', 'Wrong task')
    reward_mode = metadata.get('optimization_reward','environment')
    require(reward_mode in ('environment','strict_success'), 'Unknown optimizer reward')
    if reward_mode == 'strict_success':
        require(sha(directory/'optimization_source/train_assembly_strict.py') ==
                metadata['optimization_adapter_sha256'], 'Optimizer reward adapter changed')
    task.configure(metadata['support_profile'], metadata['controller_profile'])
    manifest = read(directory/'assembly_provenance.json')
    require(manifest == task.provenance(), 'Runtime/source/profile mismatch')
    for name, digest in manifest['sha256'].items():
        require(sha(directory/'assembly_source'/name) == digest, 'Frozen source mismatch: '+name)
    for name, digest in metadata['source_sha256'].items():
        require(sha(directory/'source'/name) == digest, 'Trainer source mismatch: '+name)
    splits = task.datasets()
    require({k:[x.key for x in v] for k,v in splits.items()} == metadata['splits'], 'Split mismatch')
    layouts = {x.key:x for v in splits.values() for x in v}
    require(sum(len(v) for v in splits.values()) == len(layouts), 'Overlapping splits')
    criterion = read(PROTOCOL)['acceptance']
    checked = {}
    records = []

    def check(row, split):
        require(row['layout_id'] in metadata['splits'][split], 'Layout outside split')
        layout = layouts[row['layout_id']]
        require(row['prompt'] == task.prompt(layout), 'Model prompt mismatch')
        if not row.get('evidence_id'):
            try:
                parse_program(row['completion'])
            except (SyntaxError, ValueError):
                require(not row['success'] and not row['valid'] and row['reward'] == 0,
                        'Invalid program received credit')
                if reward_mode == 'strict_success':
                    require(row['dense_reward'] == 0., 'Invalid program has dense credit')
                records.append(row)
                return
            raise ValueError('Valid program lacks physical evidence')
        evidence = Path(row['evidence'])
        key = row['evidence_id']
        require(evidence.name == key, 'Evidence identity/path mismatch')
        if key not in checked:
            certificate = read(evidence/'certificate.json')
            require(certificate['identity_sha256'] == key, 'Certificate identity mismatch')
            for name, digest in certificate['files'].items():
                require(sha(evidence/name) == digest, 'Evidence file changed: '+str(evidence/name))
            require(read(evidence/'provenance.json') == manifest, 'Evidence provenance mismatch')
            raw = read(evidence/'result.json')
            calls = parse_program((evidence/'program.py').read_text())
            identity = dict(seed=raw['seed'], calls=calls, provenance=manifest)
            require(hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==key,
                    'Evidence not bound to program/seed/provenance')
            require(raw['error'] != 'interrupted', 'Interrupted result certified')
            unsafe = (raw['peak_penetration_m'] > criterion['max_penetration_m'] or
                      raw['nonfinite_or_warning'] or bool(raw['unexpected_robot_contacts']))
            measured_reward = score(raw['verified_released_members'], raw['fasteners']['intact'],
                raw['preflip_stable_s'], raw['inverted_hold_s'], raw['passes'], unsafe)
            require(abs(raw['reward']-measured_reward)<1e-12, 'Physical reward mismatch')
            require(raw['success'] == raw['passes'], 'Success differs from strict acceptance')
            if raw['success']:
                require(raw['finished'] and raw['flipped'] and raw['error'] is None and not unsafe,
                        'Successful trajectory has execution/safety failure')
                require(raw['fasteners']['intact']==criterion['intact_fasteners'] and
                    raw['preflip_stable_s']+1e-8>=criterion['preflip_released_stable_s'] and
                    raw['inverted_hold_s']+1e-8>=criterion['inverted_hold_s'] and
                    raw['airborne_relative_position_peak_m']<criterion['airborne_relative_position_m'] and
                    raw['airborne_relative_angle_peak_deg']<criterion['airborne_relative_angle_deg'],
                    'Success violates recorded full-task gates')
            checked[key] = raw, calls, certificate
        raw, calls, certificate = checked[key]
        require(raw['seed'] == layout.seed, 'Reward from a different layout')
        require(row['result_sha256'] == certificate['files']['result.json'], 'Row result digest mismatch')
        requested = parse_program(row['completion'])
        if requested != calls:
            require(row['cache_reuse']=='terminal_failure_prefix' and
                task.terminal_failure_prefix(raw, requested) is not None, 'Unjustified cache reuse')
        expected = {k:v for k,v in raw.items() if k not in ('history','final_state','last_flip')}
        if row['truncated']:
            expected.update(reward=0.,success=False,valid=False,error='truncated')
        if reward_mode == 'strict_success':
            expected.update(dense_reward=expected['reward'],reward=float(expected['success']))
        for k,v in expected.items():
            compare(v,row[k],k)
        records.append(row)

    training = [json.loads(s) for s in (directory/'training.jsonl').read_text().splitlines()]
    samples = [json.loads(s) for s in (directory/'rollouts.jsonl').read_text().splitlines()]
    by_step = defaultdict(list)
    for row in samples:
        check(row,'train');by_step[row['step']].append(row)
    require([r['step'] for r in training]==list(range(1,metadata['steps']+1)), 'Incomplete updates')
    require(sorted(by_step)==[r['step'] for r in training], 'Training row coverage mismatch')
    rng = random.Random(metadata['seed'])
    for update in training:
        rows = by_step[update['step']]
        expected_layouts = [x.key for x in rng.sample(splits['train'],metadata['prompts_per_step'])
                            for _ in range(metadata['group_size'])]
        require([r['layout_id'] for r in rows]==expected_layouts, 'Unexpected optimization layouts')
        summary = metrics(rows)
        require(abs(summary['mean_reward']-update['reward'])<1e-6 and
                summary['success_rate']==update['success_rate'], 'Training aggregate mismatch')
    validation = [(0,read(directory/'validation_baseline.json'))]
    validation += [(int(p.stem.split('_')[-1]),read(p)) for p in sorted(directory.glob('validation_[0-9]*.json'))]
    expected_steps = [0]+[s for s in range(1,metadata['steps']+1) if s%metadata['eval_every']==0 or s==metadata['steps']]
    require([s for s,_ in validation]==expected_steps, 'Validation schedule mismatch')
    best, best_step = None, None
    for step, report in validation:
        require([r['layout_id'] for r in report['rows']]==metadata['splits']['validation'], 'Validation coverage mismatch')
        for row in report['rows']:check(row,'validation')
        compare(metrics(report['rows']),report['metrics'])
        candidate = report['metrics']['success_rate'],report['metrics']['mean_reward']
        if best is None or candidate>best:best,best_step=candidate,step
    selection = read(directory/'selection.json')
    require(selection['best_step']==best_step and selection['selection_split']=='validation', 'Checkpoint selection mismatch')
    require((selection['validation_success_rate'],selection['validation_mean_reward'])==best, 'Selected score mismatch')
    if best_step:
        checkpoint = directory/'checkpoints'/f'step-{best_step:04d}'
        cert = read(checkpoint/'checkpoint.json')
        for name,digest in cert['files_sha256'].items():
            require(sha(checkpoint/name)==digest, 'Checkpoint changed')
        require(sha(checkpoint/'adapter_model.safetensors')==sha(directory/'best_adapter/adapter_model.safetensors'),
                'Selected adapter differs from selected update')
    test = read(directory/'test_comparison.json')
    for key,label in [('before','baseline'),('after','trained')]:
        rows = test[key]
        require([r['layout_id'] for r in rows]==metadata['splits']['test'], 'Held-out coverage mismatch')
        for row in rows:check(row,'test')
        compare(metrics(rows),test[label])
    wins=sum(not a['success'] and b['success'] for a,b in zip(test['before'],test['after']))
    losses=sum(a['success'] and not b['success'] for a,b in zip(test['before'],test['after']))
    require(wins==test['paired_wins'] and losses==test['paired_losses'], 'Paired counts mismatch')
    require(abs((wins-losses)/len(test['before'])-test['delta_success_rate'])<1e-12,'Paired delta mismatch')
    from scipy.stats import binomtest, beta
    n=len(test['before'])
    def interval(k):
        return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
                float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
    return dict(passes=True, scope='All records, hashes, reward accounting, split isolation and validation-only selection. Independent physics replay and model regeneration are separate checks.',
        optimizer_updates=len(training),training_samples=len(samples),validation_records=sum(len(v['rows']) for _,v in validation),
        test_records=2*n,unique_certified_trajectories=len(checked),source_asset_files=len(manifest['sha256']),
        selected_step=best_step,profiles=manifest['profiles'],optimization_reward=reward_mode,
        baseline=test['baseline'],trained=test['trained'],
        paired_wins=wins,paired_losses=losses,paired_exact_two_sided_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
        baseline_success_exact_95_ci=interval(test['baseline']['successes']),trained_success_exact_95_ci=interval(test['trained']['successes']),
        limitation='Eight randomized layouts are a small simulator pilot; these intervals do not establish real-world reliability.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=audit(a.run);Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
