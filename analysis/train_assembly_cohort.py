"""Full-layout, synchronous GRPO with batched GPU inference and prefix physics.

UNRUN PROTOTYPE at the 2026-10-24 user-requested shutdown. Components have
fixture checks; the composed real-model/physics workflow still needs preflight.

All 32 training layouts contribute to every global optimizer step. All three
permutations remain learnable. Prefix reuse skips only identical measured
physical histories; complete strict outcomes remain the sole training reward.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import shutil
import time

from analysis.assembly_visual_prompt import dump,read,sha

ALGORITHM='full_layout_cohort_grpo_v1'
SOURCES=('train_assembly_cohort.py','audit_assembly_cohort.py','assembly_cohort_grpo.py',
 'assembly_pipelined_rollouts.py','assembly_prefix_rewards.py','assembly_prefix_runner.py',
 'assembly_prefix_evidence.py','assembly_memory_snapshot.py','benchmark_assembly_prefix.py',
 'assembly_training_state.py','cap_rl_assembly_grammar.py','train_assembly_strict.py')
HYPERPARAMETERS=dict(steps=4,eval_every=1,group_size=4,prompts_per_step=32,
 generation_batch=16,microbatch=8,update_epochs=8,lr=.00002,beta=.005,
 sampling_temperature=3.,seed=4177,physics_workers=6,physics_threads=1)


def layout_schedule(keys):
    rng=random.Random(HYPERPARAMETERS['seed']);schedule=[]
    for _ in range(HYPERPARAMETERS['steps']):
        row=list(keys);rng.shuffle(row);schedule.append(row)
    return schedule


def configure(metadata):
    from experiments.cap_rl import train as trainer
    from analysis import cap_rl_assembly_grammar as grammar,assembly_prefix_rewards as rewards
    grammar.install(trainer);trainer.task=trainer.get_task('assembly_full')
    task=trainer.task;task.DATASET_PROFILE='varied_v2'
    task.PHYSICS_WORKERS=metadata['physics_workers'];task.PHYSICS_THREADS=metadata['physics_threads']
    task.configure(metadata['support_profile'],metadata['controller_profile'],'vectorized_fused')
    rewards.install(task);task.EXPECTED_PROVENANCE=task.provenance()
    return trainer,grammar


def create(parent,out):
    from analysis import assembly_visual_prompt as visual
    previous=read(parent/'metadata.json');selection=read(parent/'selection.json')
    if (selection['validation_success_rate']>0.5 or
        any((parent/n).exists() for n in ('test_comparison.json','raw_test_baseline.json'))):
        raise ValueError('Requires completed non-improving development phase and untouched test')
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'process_start.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),algorithm=ALGORITHM))
    metadata={k:previous[k] for k in ('model','model_revision','max_tokens','splits',
                 'support_profile','controller_profile','initial_adapter_sha256',
                 'optimization_adapter_sha256','decoding_source_sha256','source_sha256')}
    metadata.update(HYPERPARAMETERS,task='assembly_full',assembly_split='varied_v2',
        optimizer_algorithm=ALGORITHM,assembly_backend='vectorized_fused',
        optimization_reward='strict_success',policy_decoding='assembly_permutations_v1',
        policy_action_space=24**3,policy_choice_tokens=12,prompt_profile=visual.PROFILE,
        reference_policy='initial',out=str(out),prefix_calls=9,
        warm_start_weights_sha256=sha(parent/'warm_start_adapter/adapter_model.safetensors'),
        prior_run=str(parent),prior_metadata_sha256=sha(parent/'metadata.json'),
        prior_selection_sha256=sha(parent/'selection.json'),
        selection_rule='Earliest highest greedy validation strict-success count; ties retain earlier checkpoint. Stop only at 8/8 or after four cohorts. Test only once after learned 8/8 selection; target at least 20/24 and paired improvement.',
        timing_scope='Includes fresh physical rollout waits, model generation/logprobs, global gradient updates and scheduled validation; setup and earlier phases separately recorded.',
        training_scope='128 own-policy full programs per cohort: four per each of 32 training layouts. All three order permutations learnable. Initial RGB-derived text; no hidden poses, expert plans or validation reward in gradients.')
    metadata['training_layout_schedule']=layout_schedule(metadata['splits']['train'])
    for folder in ('initial_adapter','warm_start_adapter','source','decoding_source','optimization_source'):
        shutil.copytree(parent/folder,out/folder)
    history=out/'prior_phase';history.mkdir()
    for name in ('metadata.json','selection.json','training.jsonl','validation_baseline.json','online_retirement.json','online_retirement_policy.json'):
        if (parent/name).exists():shutil.copyfile(parent/name,history/name)
    for path in parent.glob('validation_[0-9]*.json'):shutil.copyfile(path,history/path.name)
    for folder,names in (('cohort_source',SOURCES),('preference_source',visual.SOURCES)):
        destination=out/folder;destination.mkdir()
        for name in names:shutil.copyfile(Path(__file__).with_name(name),destination/name)
        metadata['cohort_source_sha256' if folder=='cohort_source' else 'visual_source_sha256']={n:sha(destination/n) for n in names}
    metadata['training_state_source_sha256']=sha(Path(__file__).with_name('assembly_training_state.py'))
    trainer,grammar=configure(metadata);manifest=trainer.task.provenance()
    dump(out/'assembly_provenance.json',manifest)
    physics_root=Path(__file__).resolve().parents[1]
    # Overlay launchers provide the actual physics repository through assembly.__file__.
    from experiments.cap_rl import assembly
    physics_root=Path(assembly.__file__).resolve().parents[2]
    for name,digest in manifest['sha256'].items():
        destination=out/'assembly_source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(physics_root/name,destination)
        if sha(destination)!=digest:raise ValueError('Physics changed during source snapshot')
    dump(out/'metadata.json',metadata)
    visual.prepare(out,metadata,'development');metadata['observations_manifest_sha256']=sha(out/'observations_manifest.json')
    dump(out/'metadata.json',metadata);dump(out/'observation_preflight.json',visual.audit_observations(out,metadata))
    return metadata,trainer,grammar


def optimize(metadata,out,trainer,grammar):
    import torch
    from transformers import set_seed
    from analysis import assembly_visual_prompt as visual,assembly_cohort_grpo as cohort
    from analysis.assembly_prefix_rewards import GroupedPrefixQueue
    from analysis.assembly_pipelined_rollouts import dispatch
    from analysis.assembly_training_state import save
    from analysis.train_assembly_strict import strict_results
    from analysis.cap_rl_assembly_records import metrics
    set_seed(metadata['seed']);torch.set_num_threads(4)
    model,tokenizer=trainer.load_model(metadata['model'],metadata['model_revision'],str(out/'warm_start_adapter'),
        trainable=True,reference_adapter=str(out/'initial_adapter'))
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=metadata['lr'])
    splits=trainer.task.datasets();by_key={x.key:x for x in splits['train']}
    started=time.monotonic();queue=GroupedPrefixQueue(trainer.task,workers=metadata['physics_workers'])
    dump(out/'memory_admission.json',queue.memory_budget)
    def generate_validation(version):
        return dispatch(trainer,model,tokenizer,splits['validation'],queue,
            policy_version=version,max_tokens=metadata['max_tokens'],sample=False,prepare_logprobs=False)
    def report(batch):
        results=strict_results(batch.collect())
        rows=trainer.rows_for(batch.layouts,batch.texts,results)
        return dict(metrics=metrics(rows),rows=rows,timing=batch.timing)
    try:
        with visual.installed(out,metadata):
            with trainer.reference_policy(model):baseline_pending=generate_validation('initial-reference')
            best_step=0;best=0;baseline=None
            for step,keys in enumerate(metadata['training_layout_schedule'],1):
                round_started=time.monotonic()
                layouts=[by_key[key] for key in keys for _ in range(metadata['group_size'])]
                chunks=[layouts[i:i+metadata['generation_batch']] for i in range(0,len(layouts),metadata['generation_batch'])]
                version=sha(out/('warm_start_adapter' if step==1 else f'checkpoints/step-{step-1:04d}')/'adapter_model.safetensors')
                batches=cohort.dispatch_all(trainer,model,tokenizer,chunks,queue,policy_version=version,
                    max_tokens=metadata['max_tokens'],temperature=metadata['sampling_temperature'],microbatch=metadata['microbatch'])
                prepared=time.monotonic()
                print(json.dumps(dict(stage='cohort_dispatched',step=step,programs=len(layouts),generation_and_logprob_seconds=prepared-round_started,policy_sha256=version)),flush=True)
                if baseline is None:
                    baseline=report(baseline_pending);dump(out/'validation_baseline.json',baseline)
                    best=baseline['metrics']['successes'];shutil.copytree(out/'initial_adapter',out/'best_adapter')
                    print(json.dumps(dict(stage='baseline_validation',**baseline['metrics'])),flush=True)
                results=[strict_results(batch.collect()) for batch in batches]
                rows=[dict(row,step=step,behavior_policy_sha256=version) for batch,values in zip(batches,results)
                      for row in trainer.rows_for(batch.layouts,batch.texts,values)]
                # Persist the sampled inputs and frozen probabilities before updating.
                rollout_path=out/f'rollout_tensors_{step:04d}.pt'
                torch.save([dict(sequences=b.sequences,attention=b.attention,mask=b.mask,
                    old=b.old,reference=b.reference,layouts=[x.key for x in b.layouts],texts=b.texts,
                    policy_version=b.policy_version) for b in batches],rollout_path)
                trainer.append(out/'rollouts.jsonl',rows)
                update_started=time.monotonic()
                epochs=cohort.update(trainer,model,optimizer,batches,results,group_size=metadata['group_size'],
                    microbatch=metadata['microbatch'],epochs=metadata['update_epochs'],beta=metadata['beta'],temperature=metadata['sampling_temperature'])
                summary=metrics(rows)
                rewards=torch.tensor([r['reward'] for r in rows]).reshape(-1,metadata['group_size'])
                metric=dict(step=step,reward=summary['mean_reward'],success_rate=summary['success_rate'],
                    active_groups=int((rewards.std(-1)>0).sum()),epoch_metrics=epochs,
                    optimizer_steps_total=step*metadata['update_epochs'],sampled_programs=len(rows),
                    unique_evidence=len({r['evidence_id'] for r in rows}),cache_hits=sum(r['cache_hit'] for r in rows),
                    batch_duplicates=sum(r['batch_duplicate'] for r in rows),
                    generation_and_logprob_seconds=prepared-round_started,
                    reward_collection_seconds=update_started-prepared,optimization_seconds=time.monotonic()-update_started,
                    step_seconds=time.monotonic()-round_started,elapsed_seconds=time.monotonic()-started,
                    rollout_tensors_sha256=sha(rollout_path),behavior_policy_sha256=version,
                    batch_timing=[b.timing for b in batches])
                trainer.append(out/'training.jsonl',[metric]);save(model,optimizer,out,metric)
                print(json.dumps(dict(stage='cohort_update',**metric)),flush=True)
                validation=report(generate_validation(f'checkpoint-{step}'));dump(out/f'validation_{step:04d}.json',validation)
                print(json.dumps(dict(stage='validation',step=step,**validation['metrics'])),flush=True)
                if validation['metrics']['successes']>best:
                    best=validation['metrics']['successes'];best_step=step;trainer.save_policy(model,out/'best_adapter')
                if best==8 and best_step>0:break
            trainer.save_policy(model,out/'last_adapter');tokenizer.save_pretrained(out/'best_adapter')
            dump(out/'selection.json',dict(best_step=best_step,validation_success_rate=best/8,
                validation_mean_reward=best/8,selection_split='validation',completed_rollout_rounds=step,
                completed_optimizer_updates=step*metadata['update_epochs'],planned_rollout_rounds=metadata['steps'],
                stopped_at_validation_ceiling=step<metadata['steps'],training_seconds=time.monotonic()-started))
    finally:queue.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out).resolve()
    metadata,trainer,grammar=create(Path(args.parent).resolve(),out)
    optimize(metadata,out,trainer,grammar)

if __name__=='__main__':main()
