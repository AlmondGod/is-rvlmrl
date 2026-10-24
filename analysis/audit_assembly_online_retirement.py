"""Pre-test audit of an explicitly retired four-round online experiment.

The record/reward/source checks are copied from the pinned historical auditor.
Only a separately verified four-round retirement replaces its perfect-validation
requirement for stopping before eight rounds. The historical auditor is retained
unchanged and still audits the earlier parent. No test evaluation is enabled.
"""
from analysis.cap_rl_assembly_records import *
from analysis import cap_rl_assembly_records as original

BASE_AUDITOR_SHA256 = '454db90ba596c56f42e3ae36bad49c12bfc54bb33cd2ab9a32fdcd5c20603fdd'


def verify_retirement(directory):
    directory=Path(directory)
    require(sha(original.__file__)==BASE_AUDITOR_SHA256,'Historical auditor changed')
    receipt=read(directory/'online_retirement.json')
    policy=read(directory/'online_retirement_policy.json')
    require(receipt['format']=='online_no_improvement_retirement_v1','Wrong retirement receipt')
    require(policy['format']=='online_no_improvement_policy_v1' and policy['boundary']==4,
            'Wrong retirement contingency')
    require(receipt['completed_rounds']==4 and receipt['optimizer_updates']==16 and
            receipt['test_evaluated'] is False,'Wrong retirement boundary')
    require(sha(directory/'online_retirement_policy.json')==receipt['policy_sha256'],
            'Retirement policy changed')
    for name,digest in policy['sources_sha256'].items():
        require(sha(directory/'online_retirement_source'/name)==digest,'Retirement source changed')
    require(policy['declared_utc'] < receipt['validation_completed_utc'],
            'Contingency must precede its decision validation')
    required={'metadata.json','training.jsonl','rollouts.jsonl','validation_baseline.json',
              'validation_0002.json','validation_0004.json','checkpoints/step-0004/checkpoint.json',
              'best_adapter/adapter_model.safetensors','last_adapter/adapter_model.safetensors'}
    require(set(receipt['files_sha256'])==required,'Incomplete retirement evidence')
    for name,digest in receipt['files_sha256'].items():
        require(sha(directory/name)==digest,'Retirement evidence changed: '+name)
    baseline=read(directory/'validation_baseline.json')['metrics']
    require(baseline['n']==8 and baseline['successes']==4,'Unexpected baseline')
    reports=[read(directory/f'validation_{step:04d}.json') for step in (2,4)]
    require(all(report['metrics']['successes']<=4 for report in reports),
            'No-improvement contingency is not satisfied')
    selected=read(directory/'selection.json')
    require(selected['best_step']==0 and selected.get('retired') is True and
            selected['completed_rollout_rounds']==4 and selected['completed_optimizer_updates']==16 and
            selected['stopped_at_validation_ceiling'] is False,'Retirement selection mismatch')
    require(sha(directory/'last_adapter/adapter_model.safetensors')==
            read(directory/'checkpoints/step-0004/checkpoint.json')['files_sha256']['adapter_model.safetensors'],
            'Final adapter differs from complete fourth checkpoint')
    require(not any((directory/name).exists() for name in ('test_comparison.json','raw_test_baseline.json',
            'unconstrained_test_comparison.json','test_observations_manifest.json')),'Test data already evaluated')
    return receipt


def _audit(directory, pretest=True):
    directory = Path(directory)
    metadata = read(directory/'metadata.json')
    online_retired = (directory/'online_retirement.json').exists()
    if online_retired:verify_retirement(directory)
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
                selected['stopped_at_validation_ceiling']==(completed_steps<8 and not online_retired),'Invalid balanced online stopping boundary')
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
    if (preference or balanced) and completed_steps<metadata['steps'] and not online_retired:
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
                    stopped_at_validation_ceiling=bool(early_stop) or ((preference or balanced) and completed_steps<metadata['steps'] and not online_retired),
                    algorithm=metadata.get('optimizer_algorithm') or 'grpo',retired=retired or online_retired,parent_grpo_audit=parent_audit,
                    unique_certified_trajectories=len(checked),
                    **({'initial_rgb_observations':observation_audit} if observation_audit is not None else {}))


def audit(directory,pretest=True):
    if not pretest:raise ValueError('This retired-phase auditor only supports pre-test checks')
    report=_audit(directory,pretest=True)
    receipt=verify_retirement(directory)
    report['online_retirement_sha256']=sha(Path(directory)/'online_retirement.json')
    report['online_retirement_policy_sha256']=receipt['policy_sha256']
    report['auditor_sha256']=sha(__file__)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();result=audit(args.run)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
