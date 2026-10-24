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


def input_receipt(directory):
    """Bind an audit to its complete set of scored records and selection inputs."""
    directory = Path(directory)
    names = ['metadata.json', 'selection.json', 'training.jsonl', 'rollouts.jsonl',
             'validation_baseline.json', 'test_comparison.json', 'raw_test_baseline.json',
             'unconstrained_validation_comparison.json', 'unconstrained_test_comparison.json',
             'validation_ceiling_policy.json', 'validation_ceiling_stop.json',
             'retirement_policy.json', 'retirement.json', 'preference_data.json', 'parent_record_audit.json',
             'initial_adapter/adapter_model.safetensors','best_adapter/adapter_model.safetensors',
             'warm_start_adapter/adapter_model.safetensors']
    names += [p.name for p in directory.glob('validation_[0-9]*.json')]
    result={name: sha(directory/name) for name in sorted(names) if (directory/name).is_file()}
    if (directory/'parent_run').is_dir():
        result.update({'parent_run/'+name:digest for name,digest in input_receipt(directory/'parent_run').items()})
    for folder in ('observations','constant_prompt_attempt','visual_prompt_attempt'):
        for path in sorted((directory/folder).rglob('*')):
            if path.is_file():result[str(path.relative_to(directory))]=sha(path)
    for name in ('observations_manifest.json','test_observations_manifest.json','observation_preflight.json'):
        if (directory/name).is_file():result[name]=sha(directory/name)
    return result


def audit_retirement(directory, metadata):
    receipt=read(directory/'retirement.json')
    policy=read(directory/'retirement_policy.json')
    require(receipt['format']=='assembly_plateau_retirement_v1' and
            receipt['completed_rounds']==6 and receipt['planned_rounds']==metadata['steps']==16 and
            receipt['optimizer_updates']==24 and receipt['test_evaluated'] is False,
            'Invalid declared plateau retirement')
    require(policy['format']=='assembly_plateau_retirement_policy_v1' and policy['boundary']==6,
            'Missing plateau contingency')
    require(sha(directory/'retirement_source/retire_assembly_plateau.py')==receipt['source_sha256']==policy['source_sha256']
            and sha(directory/'retirement_source/protocol.md')==policy['protocol_sha256'], 'Retirement source changed')
    required={'metadata.json','training.jsonl','rollouts.jsonl','validation_baseline.json','retirement_policy.json',
              'validation_0002.json','validation_0004.json','validation_0006.json'}
    require(set(receipt['files_sha256'])==required,'Incomplete retirement evidence')
    for name,digest in receipt['files_sha256'].items():require(sha(directory/name)==digest,'Retirement evidence changed')
    from analysis.retire_assembly_plateau import same_policy_results
    baseline=read(directory/'validation_baseline.json')
    require(all(same_policy_results(baseline,read(directory/f'validation_{s:04d}.json')) for s in (2,4,6)),
            'Plateau condition not met')
    return 6


def audit_preference_inputs(directory, metadata):
    from analysis.assembly_preference_data import build
    parent=directory/'parent_run'
    require(sha(parent/'metadata.json')==metadata['parent_metadata_sha256'] and
            sha(parent/'retirement.json')==metadata['parent_retirement_sha256'],'Parent run changed')
    parent_audit=audit(parent,pretest=True)
    compare(parent_audit,read(directory/'parent_record_audit.json'))
    require(sha(directory/'preference_data.json')==metadata['preference_data_sha256'],'Preference dataset changed')
    data=read(directory/'preference_data.json');rebuilt=build(parent)
    rebuilt['parent_run']=metadata['parent_original_path']
    from analysis import assembly_visual_prompt as visual
    if visual.enabled(metadata):
        rebuilt=visual.augment_preferences(rebuilt,directory,metadata)
        for name,digest in metadata['constant_prompt_attempt_sha256'].items():
            require(sha(directory/'constant_prompt_attempt'/name)==digest,'Earlier constant-prompt attempt changed')
        previous=read(directory/'constant_prompt_attempt/selection.json')
        previous_meta=read(directory/'constant_prompt_attempt/metadata.json')
        require(previous['completed_optimizer_updates']==48 and previous['validation_success_rate']<1.
                and previous_meta.get('prompt_profile','constant')=='constant'
                and previous_meta.get('optimizer_algorithm')=='offline_dpo_v1', 'Visual contingency was not met')
    compare(rebuilt,data)
    for name,digest in metadata['preference_source_sha256'].items():
        require(sha(directory/'preference_source'/name)==digest and sha(Path(__file__).with_name(name))==digest,
                'Use the frozen preference implementation: '+name)
    require(sha(directory/'warm_start_adapter/adapter_model.safetensors')==metadata['warm_start_weights_sha256'],
            'Warm-start weights changed')
    checkpoint=read(parent/'checkpoints/step-0006/checkpoint.json')
    require(checkpoint['files_sha256']['adapter_model.safetensors']==metadata['warm_start_weights_sha256'],
            'Preference phase did not start from declared GRPO checkpoint')
    require(metadata['steps']==48 and metadata['eval_every']==8 and metadata['prompts_per_step']==4 and
            metadata['group_size']==2 and metadata['update_epochs']==1 and metadata['lr']==.0001 and
            metadata['dpo_beta']==.1 and metadata['sampling_temperature']==1. and metadata['seed']==42,
            'Preference optimizer differs from declared protocol')
    return data,parent_audit


def audit_visual_grpo_inputs(directory,metadata):
    from analysis.train_assembly_visual_grpo import ALGORITHM,schedule
    parent=directory/'parent_run'
    require(metadata['optimizer_algorithm']==ALGORITHM,'Wrong balanced online algorithm')
    require(sha(parent/'metadata.json')==metadata['parent_metadata_sha256'] and
            sha(parent/'retirement.json')==metadata['parent_retirement_sha256'],'Warm-start parent changed')
    parent_audit=audit(parent,pretest=True);compare(parent_audit,read(directory/'parent_record_audit.json'))
    from analysis.assembly_visual_prompt import SOURCES
    require(set(metadata['online_source_sha256'])==set(SOURCES)|{'train_assembly_visual_grpo.py','cap_rl_assembly_records.py'},
            'Incomplete balanced online source receipt')
    require(sha(directory/'initial_adapter/adapter_model.safetensors')==sha(parent/'initial_adapter/adapter_model.safetensors'),
            'Balanced online reference differs from parent initial policy')
    for name,digest in metadata['online_source_sha256'].items():
        require(sha(directory/'preference_source'/name)==digest and sha(Path(__file__).with_name(name))==digest,
                'Use frozen balanced online source: '+name)
    require(sha(directory/'warm_start_adapter/adapter_model.safetensors')==metadata['warm_start_weights_sha256']==
            read(parent/'checkpoints/step-0006/checkpoint.json')['files_sha256']['adapter_model.safetensors'],
            'Wrong balanced online warm start')
    for folder in ('visual_prompt_attempt','constant_prompt_attempt'):
        for name,digest in metadata[folder+'_sha256'].items():
            require(sha(directory/folder/name)==digest,'Earlier unsuccessful attempt changed')
    previous=read(directory/'visual_prompt_attempt/selection.json')
    previous_meta=read(directory/'visual_prompt_attempt/metadata.json')
    from analysis.assembly_visual_prompt import enabled
    require(enabled(previous_meta) and previous_meta['optimizer_algorithm']=='offline_dpo_v1' and
            previous['completed_optimizer_updates']==48 and previous['validation_success_rate']<1.,
            'Balanced online contingency not satisfied')
    expected=dict(steps=8,eval_every=2,checkpoint_every=1,prompts_per_step=4,group_size=4,
        update_epochs=4,microbatch=8,lr=.0001,beta=.005,sampling_temperature=3.,seed=97)
    require(all(metadata.get(k)==v for k,v in expected.items()),'Balanced online hyperparameters changed')
    require(metadata['training_layout_schedule']==schedule(metadata['splits']['train'],97),'Balanced layout schedule changed')
    return parent_audit


def metrics(rows):
    return dict(n=len(rows), successes=sum(r['success'] for r in rows),
        success_rate=sum(r['success'] for r in rows)/len(rows),
        mean_reward=sum(r['reward'] for r in rows)/len(rows),
        valid_rate=sum(r['valid'] for r in rows)/len(rows))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_transfer(directory, metadata, split, check):
    path=directory/f'unconstrained_{split}_comparison.json'
    if not path.exists():return None
    report=read(path)
    require(metadata.get('policy_decoding')=='assembly_permutations_v1','Unexpected transfer evaluation')
    require(report['format']=='assembly_unconstrained_transfer_v1' and report['split']==split
            and report['evaluation_decoding']=='unrestricted_program' and report['selection_role']=='diagnostic_only',
            'Wrong transfer evaluation protocol')
    require(report['model_revision']==metadata['model_revision'],'Transfer model revision mismatch')
    for name,key in [('metadata.json','metadata_sha256'),('selection.json','selection_sha256'),
                     ('initial_adapter/adapter_model.safetensors','initial_weights_sha256'),
                     ('best_adapter/adapter_model.safetensors','selected_weights_sha256'),
                     ('evaluation_source/cap_rl_assembly_unconstrained.py','evaluator_source_sha256')]:
        require(sha(directory/name)==report[key],'Transfer source/weights changed: '+name)
    require(sha(Path(__file__).with_name('cap_rl_assembly_unconstrained.py'))==report['evaluator_source_sha256'],
            'Use the recorded transfer evaluator')
    for key,label in [('before','baseline'),('after','trained')]:
        require([r['layout_id'] for r in report[key]]==metadata['splits'][split],'Transfer split coverage mismatch')
        for row in report[key]:check(row,split)
        compare(metrics(report[key]),report[label])
    wins=sum(not a['success'] and b['success'] for a,b in zip(report['before'],report['after']))
    losses=sum(a['success'] and not b['success'] for a,b in zip(report['before'],report['after']))
    n=len(report['before'])
    require((wins,losses)==(report['paired_wins'],report['paired_losses']),'Transfer paired counts mismatch')
    require(abs((wins-losses)/n-report['delta_success_rate'])<1e-12,'Transfer paired delta mismatch')
    from scipy.stats import binomtest,beta
    def interval(k):
        return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
                float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
    return dict(passes=True,split=split,evaluation_decoding='unrestricted_program',records=2*n,
                baseline=report['baseline'],trained=report['trained'],paired_wins=wins,paired_losses=losses,
                paired_exact_two_sided_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
                baseline_success_exact_95_ci=interval(report['baseline']['successes']),
                trained_success_exact_95_ci=interval(report['trained']['successes']))


def audit(directory, pretest=False):
    directory = Path(directory)
    metadata = read(directory/'metadata.json')
    require(metadata['task'] == 'assembly_full', 'Wrong task')
    if metadata.get('init_adapter'):
        for name,digest in metadata['initial_adapter_sha256'].items():
            require(sha(directory/'initial_adapter'/name)==digest, 'Initial adapter changed')
    reward_mode = metadata.get('optimization_reward','environment')
    require(reward_mode in ('environment','strict_success'), 'Unknown optimizer reward')
    if reward_mode == 'strict_success':
        require(sha(directory/'optimization_source/train_assembly_strict.py') ==
                metadata['optimization_adapter_sha256'], 'Optimizer reward adapter changed')
    if metadata.get('policy_decoding'):
        from analysis.cap_rl_assembly_grammar import verify_source
        verify_source(directory,metadata)
    profiles = (metadata['support_profile'], metadata['controller_profile'])
    task.configure(*profiles, **({'assembly_backend': metadata['assembly_backend']}
                                if 'assembly_backend' in metadata else {}))
    manifest = read(directory/'assembly_provenance.json')
    require(manifest == task.provenance(), 'Runtime/source/profile mismatch')
    for name, digest in manifest['sha256'].items():
        require(sha(directory/'assembly_source'/name) == digest, 'Frozen source mismatch: '+name)
    for name, digest in metadata['source_sha256'].items():
        require(sha(directory/'source'/name) == digest, 'Trainer source mismatch: '+name)
    task.DATASET_PROFILE = metadata.get('assembly_split', 'pilot_v1')
    splits = task.datasets()
    require({k:[x.key for x in v] for k,v in splits.items()} == metadata['splits'], 'Split mismatch')
    layouts = {x.key:x for v in splits.values() for x in v}
    require(sum(len(v) for v in splits.values()) == len(layouts), 'Overlapping splits')
    criterion = read(PROTOCOL)['acceptance']
    from analysis import assembly_visual_prompt as visual
    observation_audit=visual.audit_observations(directory,metadata,include_test=not pretest) if visual.enabled(metadata) else None
    checked = {}
    records = []

    def check(row, split):
        require(row['layout_id'] in metadata['splits'][split], 'Layout outside split')
        layout = layouts[row['layout_id']]
        require(row['prompt'] == visual.expected_prompt(directory,metadata,layout), 'Model prompt mismatch')
        if not row.get('evidence_id'):
            if row.get('static_rejection'):
                from analysis.train_assembly_strict import required_call_failure
                require(reward_mode=='strict_success' and metadata.get('static_strict_rejection')=='strict_necessary_calls_v1',
                        'Static rejection is restricted to the declared strict optimizer')
                require(sha(Path(__file__).with_name('train_assembly_strict.py'))==metadata['optimization_adapter_sha256'],
                        'Use the recorded strict rejection proof implementation')
                require(required_call_failure(row['completion'])==row['static_rejection'], 'Invalid necessary-call proof')
                require(row.get('physics_started') is False and row['error']=='strict_missing_required_calls'
                        and row['valid'] and not row['success'] and row['reward']==0
                        and row['dense_reward'] is None and row.get('dense_reward_measured') is False,
                        'Static proof received unsupported physics credit')
                records.append(row)
                return
            if row.get('truncated'):
                require(row.get('physics_started') is False and row['error'] == 'truncated'
                        and not row['success'] and not row['valid'] and row['reward'] == 0,
                        'Truncated unexecuted program received credit')
                if reward_mode == 'strict_success':
                    require(row['dense_reward'] == 0., 'Truncated program has dense credit')
                records.append(row)
                return
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
    preference = metadata.get('optimizer_algorithm')=='offline_dpo_v1'
    balanced=metadata.get('optimizer_algorithm')=='grpo_visual_balanced_v1'
    require(metadata.get('optimizer_algorithm') in (None,'offline_dpo_v1','grpo_visual_balanced_v1'),'Unknown optimizer algorithm')
    completed_steps = metadata['steps']
    early_stop = None
    retired=(directory/'retirement.json').exists()
    preference_data=parent_audit=None
    if retired:
        require(not preference,'Preference run cannot use GRPO retirement')
        completed_steps=audit_retirement(directory,metadata)
    if preference:
        preference_data,parent_audit=audit_preference_inputs(directory,metadata)
        selection=read(directory/'selection.json')
        completed_steps=selection['completed_optimizer_updates']
        require(0<completed_steps<=48 and completed_steps%8==0 and selection['planned_optimizer_updates']==48,
                'Invalid preference stopping boundary')
        require(selection['stopped_at_validation_ceiling']==(completed_steps<48),'Preference early stop lacks ceiling rule')
    if balanced:
        parent_audit=audit_visual_grpo_inputs(directory,metadata)
        selected=read(directory/'selection.json');completed_steps=selected['completed_rollout_rounds']
        require(completed_steps in (2,4,6,8) and selected['planned_rollout_rounds']==8 and
                selected['completed_optimizer_updates']==completed_steps*4 and
                selected['stopped_at_validation_ceiling']==(completed_steps<8),'Invalid balanced online stopping boundary')
    if (directory/'validation_ceiling_stop.json').exists():
        early_stop = read(directory/'validation_ceiling_stop.json')
        require(reward_mode == 'strict_success', 'Ceiling stop requires binary success reward')
        require(early_stop['format'] in ('cap_rl_validation_ceiling_stop_v1','cap_rl_validation_ceiling_stop_v2') and
                early_stop['upper_bound'] == [1.,1.] and
                early_stop['test_evaluated'] is False, 'Invalid ceiling stopping rule')
        modern=early_stop['format']=='cap_rl_validation_ceiling_stop_v2'
        source='selection_source/finalize_assembly_ceiling.py' if modern else 'selection_source/stop_assembly04_at_validation_ceiling.py'
        if modern:
            require(early_stop['source_path']==source,'Unexpected selection finalizer path')
        require(sha(directory/source)==early_stop['source_sha256'],'Selection finalizer source changed')
        completed_steps=early_stop['completed_rollout_rounds' if modern else 'completed_updates']
        if modern:
            require(early_stop['optimizer_updates']==completed_steps*metadata.get('update_epochs',1),'Stopping optimizer count mismatch')
        require(early_stop['planned_rollout_rounds' if modern else 'planned_updates'] == metadata['steps'] and
                0 < completed_steps < metadata['steps'] and
                completed_steps % metadata['eval_every'] == 0, 'Invalid early-stop boundary')
        required = {'metadata.json','training.jsonl','rollouts.jsonl','validation_ceiling_policy.json',
                    'validation_baseline.json'}
        required.update(f'validation_{s:04d}.json' for s in range(metadata['eval_every'],completed_steps+1,metadata['eval_every']))
        require(set(early_stop['files_sha256']) == required, 'Incomplete stopping evidence')
        for name,digest in early_stop['files_sha256'].items():
            require(sha(directory/name) == digest, 'Stopping evidence changed: '+name)
    samples = [json.loads(s) for s in (directory/'rollouts.jsonl').read_text().splitlines()]
    by_step = defaultdict(list)
    for row in samples:
        check(row,'train');by_step[row['step']].append(row)
    require([r['step'] for r in training]==list(range(1,completed_steps+1)), 'Incomplete updates')
    require(sorted(by_step)==[r['step'] for r in training], 'Training row coverage mismatch')
    rng = random.Random(metadata['seed'])
    epochs=metadata.get('update_epochs',1)
    require(type(epochs) is int and epochs>=1,'Invalid optimizer epoch count')
    for update in training:
        if preference:
            from analysis.train_assembly_preferences import sample_pairs
            pair_ids=sample_pairs(preference_data,rng,metadata['prompts_per_step'])
            require(update['pair_indices']==pair_ids and update['optimizer_steps_total']==update['step'] and
                    update['new_training_physics_episodes']==0,'Preference optimizer accounting mismatch')
            original=[json.loads(s) for s in (directory/'parent_run/rollouts.jsonl').read_text().splitlines()]
            expected=[]
            for pair_id in pair_ids:
                pair=preference_data['pairs'][pair_id]
                for role in ('chosen','rejected'):
                    expected.append(dict(original[pair[role+'_row']],prompt=pair['prompt'],step=update['step'],preference_pair_id=pair_id,
                        preference_role=role,parent_source_row=pair[role+'_row'],offline_replay=True))
            compare(expected,by_step[update['step']])
            require(update['replayed_programs']==len(expected),'Replayed sample count mismatch')
            continue
        if 'update_epochs' in metadata:
            require(update['optimizer_steps_total']==update['step']*epochs,'Optimizer step count mismatch')
            require([x['epoch'] for x in update['epoch_metrics']]==list(range(1,epochs+1)),
                    'Optimizer epoch coverage mismatch')
        rows = by_step[update['step']]
        expected_layouts = ([key for key in metadata['training_layout_schedule'][update['step']-1]
                             for _ in range(metadata['group_size'])] if balanced else
                            [x.key for x in rng.sample(splits['train'],metadata['prompts_per_step'])
                             for _ in range(metadata['group_size'])])
        require([r['layout_id'] for r in rows]==expected_layouts, 'Unexpected optimization layouts')
        summary = metrics(rows)
        require(abs(summary['mean_reward']-update['reward'])<1e-6 and
                summary['success_rate']==update['success_rate'], 'Training aggregate mismatch')
    validation = [(0,read(directory/'validation_baseline.json'))]
    validation += [(int(p.stem.split('_')[-1]),read(p)) for p in sorted(directory.glob('validation_[0-9]*.json'))]
    expected_steps = [0]+[s for s in range(1,completed_steps+1) if s%metadata['eval_every']==0 or s==completed_steps]
    require([s for s,_ in validation]==expected_steps, 'Validation schedule mismatch')
    best, best_step = None, None
    for step, report in validation:
        require([r['layout_id'] for r in report['rows']]==metadata['splits']['validation'], 'Validation coverage mismatch')
        for row in report['rows']:check(row,'validation')
        compare(metrics(report['rows']),report['metrics'])
        candidate = report['metrics']['success_rate'],report['metrics']['mean_reward']
        if best is None or candidate>best:best,best_step=candidate,step
    selection = read(directory/'selection.json')
    if early_stop:
        require(best == (1.,1.) and early_stop['selected_step'] == best_step,
                'Early stop lacks the maximum validation score')
    if retired:require(best_step==0 and selection.get('retired') is True,'Retired unchanged run selected new weights')
    if (preference or balanced) and completed_steps<metadata['steps']:
        require(best==(1.,1.) and best_step==completed_steps,'Preference early stop lacks first perfect validation')
    require(selection['best_step']==best_step and selection['selection_split']=='validation', 'Checkpoint selection mismatch')
    require((selection['validation_success_rate'],selection['validation_mean_reward'])==best, 'Selected score mismatch')
    if best_step:
        checkpoint = directory/'checkpoints'/f'step-{best_step:04d}'
        cert = read(checkpoint/'checkpoint.json')
        for name,digest in cert['files_sha256'].items():
            require(sha(checkpoint/name)==digest, 'Checkpoint changed')
        require(sha(checkpoint/'adapter_model.safetensors')==sha(directory/'best_adapter/adapter_model.safetensors'),
                'Selected adapter differs from selected update')
    elif metadata.get('init_adapter'):
        require(sha(directory/'initial_adapter/adapter_model.safetensors') ==
                sha(directory/'best_adapter/adapter_model.safetensors'),
                'Selected baseline differs from the initial adapter')
    transfer_validation=audit_transfer(directory,metadata,'validation',check)
    if pretest:
        require(not any((directory/name).exists() for name in ('test_comparison.json','raw_test_baseline.json','unconstrained_test_comparison.json')), 'Test already evaluated')
        return dict(passes=True,scope='Pre-test source, training, validation, reward, stopping and selected-weight audit; no test outcomes read.',
                    rollout_rounds=0 if preference else len(training),planned_rollout_rounds=0 if preference else metadata['steps'],
                    optimizer_updates=len(training)*epochs,planned_optimizer_updates=metadata['steps']*epochs,
                    selected_step=best_step,validation_score=list(best),unconstrained_validation=transfer_validation,
                    stopped_at_validation_ceiling=bool(early_stop) or ((preference or balanced) and completed_steps<metadata['steps']),
                    algorithm=metadata.get('optimizer_algorithm') or 'grpo',retired=retired,parent_grpo_audit=parent_audit,
                    unique_certified_trajectories=len(checked),
                    **({'initial_rgb_observations':observation_audit} if observation_audit is not None else {}))
    test = read(directory/'test_comparison.json')
    for key,label in [('before','baseline'),('after','trained')]:
        rows = test[key]
        require([r['layout_id'] for r in rows]==metadata['splits']['test'], 'Held-out coverage mismatch')
        for row in rows:check(row,'test')
        compare(metrics(rows),test[label])
    raw_baseline=None
    if (directory/'raw_test_baseline.json').exists():
        raw=read(directory/'raw_test_baseline.json')
        require([r['layout_id'] for r in raw['rows']]==metadata['splits']['test'],'Raw baseline coverage mismatch')
        require(raw['model_revision']==metadata['model_revision'],'Raw model revision mismatch')
        for row in raw['rows']:check(row,'test')
        compare(metrics(raw['rows']),raw['metrics'])
        raw_baseline=raw['metrics']
    transfer_test=audit_transfer(directory,metadata,'test',check)
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
        input_files_sha256=input_receipt(directory),
        rollout_rounds=0 if preference else len(training),planned_rollout_rounds=0 if preference else metadata['steps'],
                    optimizer_updates=len(training)*epochs,planned_optimizer_updates=metadata['steps']*epochs,
        stopped_at_validation_ceiling=bool(early_stop) or ((preference or balanced) and completed_steps<metadata['steps']),
        training_samples=0 if preference else len(samples),replayed_training_programs=len(samples) if preference else 0,
        algorithm=metadata.get('optimizer_algorithm') or 'grpo',parent_grpo_audit=parent_audit,retired=retired,
        validation_records=sum(len(v['rows']) for _,v in validation),
        test_records=2*n,raw_test_records=n if raw_baseline is not None else 0,unique_certified_trajectories=len(checked),source_asset_files=len(manifest['sha256']),
        selected_step=best_step,profiles=manifest['profiles'],optimization_reward=reward_mode,
        baseline=test['baseline'],trained=test['trained'],raw_baseline=raw_baseline,
        unconstrained_validation=transfer_validation,unconstrained_test=transfer_test,
        paired_wins=wins,paired_losses=losses,paired_exact_two_sided_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
        baseline_success_exact_95_ci=interval(test['baseline']['successes']),trained_success_exact_95_ci=interval(test['trained']['successes']),
        limitation=f'{n} randomized layouts are a simulator pilot; these intervals do not establish real-world reliability.',
        **({'initial_rgb_observations':observation_audit} if observation_audit is not None else {}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--pretest',action='store_true')
    a=p.parse_args();result=audit(a.run,pretest=a.pretest);Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
