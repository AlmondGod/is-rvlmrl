"""Summarize completed physical assembly RL, separating samples from executions."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--retired-pilot', action='append', default=[],
                   help='Separate exploratory pilot; repeat for multiple runs. Never added to final-run updates.')
    p.add_argument('--verification-status', help='Completed supervisor status for end-to-end allocation time')
    args = p.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata, selection, test = [read(run / name) for name in
                                ('metadata.json', 'selection.json', 'test_comparison.json')]
    training = [json.loads(s) for s in (run/'training.jsonl').read_text().splitlines()]
    rollouts = [json.loads(s) for s in (run/'rollouts.jsonl').read_text().splitlines()]
    validation = [(0, read(run/'validation_baseline.json'))]
    validation.extend((int(path.stem.split('_')[-1]), read(path))
                      for path in sorted(run.glob('validation_[0-9]*.json')))
    records = rollouts + [row for _, v in validation for row in v['rows']]
    raw=read(run/'raw_test_baseline.json') if (run/'raw_test_baseline.json').exists() else None
    records += test['before'] + test['after'] + (raw['rows'] if raw else [])
    transfers = {}
    for split in ('validation', 'test'):
        path = run/f'unconstrained_{split}_comparison.json'
        if path.exists():
            transfer = read(path)
            records += transfer['before'] + transfer['after']
            transfers[split] = {key: transfer[key] for key in (
                'evaluation_decoding', 'selection_role', 'baseline', 'trained',
                'paired_wins', 'paired_losses', 'delta_success_rate', 'scope')}
    unique = {r['evidence_id']: r for r in records if r.get('evidence_id')}
    failures = lambda rows: dict(Counter(r.get('error') or 'strict_acceptance_failure'
                                        for r in rows if not r['success']))
    summary = dict(model=metadata['model'], model_revision=metadata['model_revision'],
        support_profile=metadata.get('support_profile','contact_only'),
        controller_profile=metadata.get('controller_profile','original'),
        optimization_reward=metadata.get('optimization_reward','environment'),
        task=metadata['task'], gpu=metadata['gpu'], rollout_rounds=len(training),
        optimizer_updates=len(training)*metadata.get('update_epochs',1),
        policy_decoding=metadata.get('policy_decoding','unrestricted_program'),
        policy_action_space=metadata.get('policy_action_space'),
        unconstrained_transfer=transfers,
        sampled_training_programs=len(rollouts), training_wall_seconds=selection['training_seconds'],
        allocated_training_gpu_hours=selection['training_seconds']/3600,
        compute_scope=selection.get('training_seconds_scope',
            'One GPU allocated during training including CPU physics and periodic validation; excludes setup, baseline validation, final test and audits.'),
        planned_rollout_rounds=metadata['steps'],planned_optimizer_updates=metadata['steps']*metadata.get('update_epochs',1),
        stopped_at_validation_ceiling=(run/'validation_ceiling_stop.json').exists(),
        selected_step=selection['best_step'], validation=selection,
        selected_checkpoint_elapsed_seconds=next(
            (r['elapsed_seconds'] for r in training if r['step']==selection['best_step']), 0.),
        baseline=test['baseline'], trained=test['trained'], raw_baseline=raw['metrics'] if raw else None,paired_wins=test['paired_wins'],
        paired_losses=test['paired_losses'], delta_success_rate=test['delta_success_rate'],
        paired_bootstrap_95_ci=test['paired_bootstrap_95_ci'],
        total_scored_records=len(records), distinct_measured_evidence_ids=len(unique),
        prior_record_cache_hits=sum(bool(r.get('cache_hit')) for r in records),
        within_batch_duplicates=sum(bool(r.get('batch_duplicate')) for r in records),
        terminal_failure_reuses=sum(r.get('cache_reuse')=='terminal_failure_prefix' for r in records),
        before_failure_counts=failures(test['before']), after_failure_counts=failures(test['after']),
        physically_successful_training_samples=sum(r['success'] for r in rollouts),
        active_group_updates=sum(r['active_groups']>0 for r in training),
        sampled_unique_programs=len({r['completion'] for r in rollouts}),
        statically_rejected_samples=sum(bool(r.get('static_rejection')) for r in rollouts),
        generation_seconds=sum(r.get('generation_s',0.) for r in training),
        logprob_seconds=sum(r.get('logprob_s',0.) for r in training),
        exposed_simulation_wait_seconds=sum(r.get('reward_wait_after_preparation_s',0.) for r in training),
        exposed_simulation_wait_scope='Full environment reward wait after overlapping old/reference log-probabilities; includes physics, rendering, perception and scripted control. Not a pure MuJoCo kernel time.',
        optimization_seconds=sum(r.get('optimization_seconds',0.) for r in training),
        full_visual_language_policy=False, learned_motor_controller=False,
        task_prompt_identical_across_layouts=True, physical_validation=False)
    summary['contact_only_shadow'] = {
        label: dict(n=len(test[key]), successes=sum(bool(r.get('contact_only_acceptance',{}).get('passes'))
                                                  for r in test[key]))
        for label,key in [('before','before'),('after','after')]}
    audit = read(run/'record_audit.json') if (run/'record_audit.json').exists() else None
    if audit:
        for name in ('paired_exact_two_sided_p','baseline_success_exact_95_ci','trained_success_exact_95_ci'):
            summary[name] = audit[name]
        for split, transfer in transfers.items():
            checked = audit.get('unconstrained_'+split)
            if checked:
                for name in ('paired_exact_two_sided_p', 'baseline_success_exact_95_ci',
                             'trained_success_exact_95_ci'):
                    transfer[name] = checked[name]
        summary['uncertainty_note'] = (
            'Exact success-rate intervals describe this small layout sample. '
            'The retained empirical paired bootstrap interval can collapse at all wins '
            'and should not be interpreted as certainty about broader reliability.')
    if args.verification_status:
        status = read(Path(args.verification_status))
        if status['state'] != 'complete':
            raise ValueError('Verification is not complete')
        start = read(run/'process_start.json')
        seconds = (datetime.fromisoformat(status['utc'])-
                   datetime.fromisoformat(start['started_utc'])).total_seconds()
        if seconds < 0:
            raise ValueError('Invalid process/verification timestamps')
        summary['end_to_end_allocated_gpu_hours'] = seconds/3600
        summary['end_to_end_timing_scope'] = (
            'One continuously allocated GPU from trainer process start through completed '
            'test, record audit, model regeneration and independent physical replay. '
            'Includes CPU simulation waits; excludes earlier setup, exploration and '
            'prefetched physical evidence produced before this run.')
        summary['verification_stages'] = status['completed_stages']
    summary['retired_exploratory_pilots'] = []
    for pilot in args.retired_pilot:
        retired = read(Path(pilot)/'retired.json')
        pilot_metadata = read(Path(pilot)/'metadata.json')
        pilot_metrics = retired['last_metric']
        rounds = retired['completed_updates']
        summary['retired_exploratory_pilots'].append(dict(
            run=Path(pilot).name,
            rollout_rounds=rounds,
            optimizer_updates=pilot_metrics.get('optimizer_steps_total',
                rounds*pilot_metadata.get('update_epochs', 1)),
            training_loop_gpu_hours=pilot_metrics['elapsed_seconds']/3600,
            process_gpu_hours=retired['trainer_elapsed_seconds']/3600,
            weights_used_to_initialize_final_run=False,
            reason=retired['reason']))
    (out/'summary.json').write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    steps = [row['step'] for row in training]
    axes[0].plot(steps, [100*r['success_rate'] for r in training], alpha=.5, label='Sampled training success')
    axes[0].plot([s for s, _ in validation], [100*v['metrics']['success_rate'] for _, v in validation],
                 'o-', color='#087e8b', label='Greedy validation success')
    axes[0].axvline(selection['best_step'], color='#777777', ls=':', label='Selected checkpoint')
    axes[0].set(xlabel='Rollout round (each followed by optimizer epochs)', ylabel='Strict physical success (%)', ylim=(-3, 110), title='Learning and selection')
    axes[0].legend(fontsize=8)
    values = [test['baseline'], test['trained']]
    labels=['Prior policy','Selected policy'];colors=['#b1b9c4','#087e8b']
    if raw:
        values=[raw['metrics']]+values;labels=['Raw Qwen']+labels;colors=['#d8dde5']+colors
    axes[1].bar(labels,[100*x['success_rate'] for x in values],color=colors,width=.6)
    intervals = ([audit['baseline_success_exact_95_ci'], audit['trained_success_exact_95_ci']]
                 if audit else None)
    if intervals and raw:
        from scipy.stats import beta
        k,n=raw['metrics']['successes'],raw['metrics']['n']
        intervals=[[float(beta.ppf(.025,k,n-k+1)) if k else 0.,
                    float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]]+intervals
    if intervals:
        rates = [100*x['success_rate'] for x in values]
        axes[1].errorbar(list(range(len(values))), rates,
            yerr=[[v-100*c[0] for v,c in zip(rates,intervals)],
                  [100*c[1]-v for v,c in zip(rates,intervals)]],
            fmt='o', color='#253343', markersize=4, capsize=5,
            label='Exact 95% success-rate interval')
        axes[1].legend(loc='upper left', fontsize=8)
    for i, item in enumerate(values):
        label_height = 100*(intervals[i][1] if intervals else item['success_rate'])+3
        axes[1].text(i, label_height,
                     f"{item['successes']}/{item['n']} ({100*item['success_rate']:.1f}%)", ha='center')
    axes[1].set(ylim=(0, 120), ylabel='Strict physical success (%)', title='Paired held-out layouts')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Qwen3.5-4B | Three-arm frame assembly, fastening and flip', fontsize=13)
    fig.savefig(out/'qwen35-three-arm-results.png', dpi=180)
    plt.close(fig)
    if 'test' in transfers:
        transfer = transfers['test']
        fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
        labels = ['Prior\nconstrained', 'Selected\nconstrained',
                  'Prior\nfree-form', 'Selected\nfree-form']
        values = [test['baseline'], test['trained'], transfer['baseline'], transfer['trained']]
        rates = [100*item['success_rate'] for item in values]
        ax.bar(labels, rates, color=['#b1b9c4', '#087e8b', '#b1b9c4', '#8a60a6'], width=.65)
        intervals = None
        if audit and 'baseline_success_exact_95_ci' in transfer:
            intervals = [audit['baseline_success_exact_95_ci'], audit['trained_success_exact_95_ci'],
                         transfer['baseline_success_exact_95_ci'], transfer['trained_success_exact_95_ci']]
            ax.errorbar(range(4), rates,
                yerr=[[v-100*c[0] for v,c in zip(rates,intervals)],
                      [100*c[1]-v for v,c in zip(rates,intervals)]],
                fmt='o', color='#253343', markersize=4, capsize=5,
                label='Exact 95% success-rate interval')
            ax.legend(loc='upper left', fontsize=8)
        for i, item in enumerate(values):
            height = 100*(intervals[i][1] if intervals else item['success_rate'])+3
            ax.text(i, height, f"{item['successes']}/{item['n']}", ha='center')
        ax.axvline(1.5, color='#dddddd')
        ax.set(ylim=(0, 120), ylabel='Strict physical success (%)',
               title='Same frozen weights and held-out layouts: constrained and free-form decoding')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
        fig.savefig(out/'qwen35-decoding-transfer.png', dpi=180)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
