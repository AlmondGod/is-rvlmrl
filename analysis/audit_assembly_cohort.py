"""Audit full-layout cohort training, physical evidence and frozen selection.

UNRUN PROTOTYPE at the 2026-10-24 user-requested shutdown. Components have
fixture checks; the composed real-model/physics workflow still needs preflight.

Record audit is separate from independent model regeneration and fresh physics.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from analysis.cap_rl_assembly_records import read,sha,require,metrics,input_receipt
from experiments.cap_rl import task_assembly as task
from experiments.cap_rl.assembly import parse_program,score,PROTOCOL
from experiments.cap_rl.audit_assembly import compare


def make_checker(directory,metadata,manifest):
    from analysis import assembly_visual_prompt as visual
    criterion=read(PROTOCOL)['acceptance'];reward_mode='strict_success'
    layouts={x.key:x for values in task.datasets().values() for x in values}
    checked={};records=[]
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
            from analysis.assembly_prefix_rewards import verify_episode_prefix
            verify_episode_prefix(evidence,manifest['execution_prefix'])
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

    return check,checked,records

def audit(directory,pretest=True):
    from analysis.train_assembly_cohort import ALGORITHM,HYPERPARAMETERS,SOURCES,configure,layout_schedule
    from analysis import assembly_visual_prompt as visual
    directory=Path(directory);metadata=read(directory/'metadata.json')
    require(metadata['optimizer_algorithm']==ALGORITHM,'Unknown cohort algorithm')
    require(all(metadata.get(k)==v for k,v in HYPERPARAMETERS.items()),'Cohort hyperparameters changed')
    require(metadata['policy_action_space']==24**3 and metadata['policy_choice_tokens']==12 and
            metadata['prefix_calls']==9 and metadata['optimization_reward']=='strict_success','Wrong action/reward scope')
    require(set(metadata['cohort_source_sha256'])==set(SOURCES),'Missing cohort implementation receipt')
    for name,digest in metadata['cohort_source_sha256'].items():
        require(sha(Path(__file__).with_name(name))==digest==sha(directory/'cohort_source'/name),'Changed cohort implementation: '+name)
    for name,digest in metadata['initial_adapter_sha256'].items():
        require(sha(directory/'initial_adapter'/name)==digest,'Initial adapter changed')
    require(sha(directory/'warm_start_adapter/adapter_model.safetensors')==metadata['warm_start_weights_sha256'],'Warm start changed')
    require(sha(directory/'prior_phase/metadata.json')==metadata['prior_metadata_sha256'] and
            sha(directory/'prior_phase/selection.json')==metadata['prior_selection_sha256'],'Prior phase changed')
    prior=read(directory/'prior_phase/metadata.json');selection=read(directory/'prior_phase/selection.json')
    require(selection['validation_success_rate']<=.5 and prior['warm_start_weights_sha256']==metadata['warm_start_weights_sha256'],
            'New phase contingency or warm start mismatch')
    trainer,grammar=configure(metadata);grammar.verify_source(directory,metadata)
    manifest=read(directory/'assembly_provenance.json')
    require(manifest==task.provenance(),'Physical/prefix runtime differs')
    for name,digest in manifest['sha256'].items():
        require(sha(directory/'assembly_source'/name)==digest,'Physics snapshot changed: '+name)
    from experiments.cap_rl import train
    for name,digest in metadata['source_sha256'].items():
        require(sha(directory/'source'/name)==digest,'Trainer snapshot changed')
    require({k:[x.key for x in v] for k,v in task.datasets().items()}==metadata['splits'],'Split definition changed')
    require(len(set(sum(metadata['splits'].values(),[])))==64,'Overlapping/incomplete splits')
    observations=visual.audit_observations(directory,metadata,include_test=not pretest)
    require(metadata['training_layout_schedule']==layout_schedule(metadata['splits']['train']),'Training order changed')
    selection=read(directory/'selection.json');rounds=selection['completed_rollout_rounds']
    require(0<rounds<=metadata['steps'] and selection['planned_rollout_rounds']==metadata['steps'] and
            selection['completed_optimizer_updates']==rounds*metadata['update_epochs'],'Optimizer count mismatch')
    check,checked,records=make_checker(directory,metadata,manifest)
    training=[json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
    samples=[json.loads(line) for line in (directory/'rollouts.jsonl').read_text().splitlines()]
    require([m['step'] for m in training]==list(range(1,rounds+1)),'Incomplete training metrics')
    by_step=defaultdict(list)
    for row in samples:check(row,'train');by_step[row['step']].append(row)
    require(sorted(by_step)==list(range(1,rounds+1)),'Incomplete sampled programs')
    for metric in training:
        step=metric['step'];rows=by_step[step];expected=[key for key in metadata['training_layout_schedule'][step-1] for _ in range(4)]
        require([r['layout_id'] for r in rows]==expected,'Cohort must contain four own samples from all 32 layouts')
        behavior=sha(directory/('warm_start_adapter' if step==1 else f'checkpoints/step-{step-1:04d}')/'adapter_model.safetensors')
        require(metric['behavior_policy_sha256']==behavior and all(r['behavior_policy_sha256']==behavior for r in rows),'Wrong sampled policy')
        summary=metrics(rows)
        require(metric['reward']==summary['mean_reward'] and metric['success_rate']==summary['success_rate'] and
                metric['sampled_programs']==128 and metric['optimizer_steps_total']==step*8,'Training aggregate mismatch')
        require([m['epoch'] for m in metric['epoch_metrics']]==list(range(1,9)) and
                all(m['sampled_programs']==128 for m in metric['epoch_metrics']),'Partial-cohort update')
        tensors=directory/f'rollout_tensors_{step:04d}.pt'
        require(sha(tensors)==metric['rollout_tensors_sha256'],'Frozen probability payload changed')
        checkpoint=directory/f'checkpoints/step-{step:04d}';cert=read(checkpoint/'checkpoint.json')
        compare(cert['metric'],metric)
        require(cert['training_state_source_sha256']==metadata['training_state_source_sha256'] and
                'training_state.pt' in cert['files_sha256'],'Missing optimizer/RNG checkpoint')
        for name,digest in cert['files_sha256'].items():require(sha(checkpoint/name)==digest,'Checkpoint changed')
    validation=[(0,read(directory/'validation_baseline.json'))]+[(int(p.stem.split('_')[-1]),read(p))
                for p in sorted(directory.glob('validation_[0-9]*.json'))]
    require([s for s,_ in validation]==list(range(rounds+1)),'Validation schedule mismatch')
    best=-1;best_step=None
    for step,report in validation:
        require([r['layout_id'] for r in report['rows']]==metadata['splits']['validation'],'Validation coverage mismatch')
        for row in report['rows']:check(row,'validation')
        compare(metrics(report['rows']),report['metrics'])
        if report['metrics']['successes']>best:best=report['metrics']['successes'];best_step=step
    require(selection['best_step']==best_step and selection['selection_split']=='validation' and
            selection['validation_success_rate']==selection['validation_mean_reward']==best/8,'Selection mismatch')
    require(selection['stopped_at_validation_ceiling']==(rounds<4),'Stopping flag mismatch')
    if rounds<4:require(best==8 and best_step==rounds,'Early stop without first perfect validation')
    selected=directory/('initial_adapter' if best_step==0 else f'checkpoints/step-{best_step:04d}')/'adapter_model.safetensors'
    require(sha(selected)==sha(directory/'best_adapter/adapter_model.safetensors'),'Selected weights mismatch')
    report=dict(passes=True,algorithm=ALGORITHM,training_samples=len(samples),rollout_rounds=rounds,
        optimizer_updates=rounds*8,selected_step=best_step,validation_successes=best,
        unique_certified_trajectories=len(checked),initial_rgb_observations=observations,
        scope='Source, full-cohort records, strict rewards, prefix origin evidence, splits and validation selection. Fresh physics and model regeneration are separate checks.')
    if pretest:
        require(not (directory/'test_comparison.json').exists(),'Held-out test already evaluated')
        return report
    require(best==8 and best_step>0,'Held-out test before declared validation gate')
    test=read(directory/'test_comparison.json')
    for key,label in (('before','baseline'),('after','trained')):
        require([r['layout_id'] for r in test[key]]==metadata['splits']['test'],'Test coverage mismatch')
        for row in test[key]:check(row,'test')
        compare(metrics(test[key]),test[label])
    wins=sum(not a['success'] and b['success'] for a,b in zip(test['before'],test['after']))
    losses=sum(a['success'] and not b['success'] for a,b in zip(test['before'],test['after']))
    require((wins,losses)==(test['paired_wins'],test['paired_losses']),'Paired counts mismatch')
    require(test['delta_success_rate']==(wins-losses)/24,'Paired delta mismatch')
    from scipy.stats import binomtest,beta
    def interval(k):return [float(beta.ppf(.025,k,25-k)) if k else 0.,float(beta.ppf(.975,k+1,24-k)) if k<24 else 1.]
    report.update(baseline=test['baseline'],trained=test['trained'],paired_wins=wins,paired_losses=losses,
        paired_exact_two_sided_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
        baseline_success_exact_95_ci=interval(test['baseline']['successes']),
        trained_success_exact_95_ci=interval(test['trained']['successes']),
        input_files_sha256=input_receipt(directory))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--include-test',action='store_true');a=p.parse_args()
    result=audit(a.run,not a.include_test);Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
