"""Balanced on-policy GRPO after the camera-conditioned replay experiment.

Every declared training layout receives four own-policy samples once. Full
physical outcomes, frozen old probabilities and the initial reference drive
four clipped updates per rollout round. No validation reward enters training.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import time

from analysis.assembly_visual_prompt import read,sha,dump

ALGORITHM='grpo_visual_balanced_v1'


def schedule(keys, seed=97):
    if len(keys)!=32 or len(set(keys))!=32:raise ValueError('Expected 32 unique training layouts')
    values=list(keys);random.Random(seed).shuffle(values)
    return [values[i:i+4] for i in range(0,32,4)]


def snapshot_attempt(source,destination):
    destination.mkdir()
    for name in ('metadata.json','selection.json','training.jsonl','rollouts.jsonl','validation_baseline.json','process_start.json'):
        shutil.copyfile(source/name,destination/name)
    for path in source.glob('validation_[0-9]*.json'):shutil.copyfile(path,destination/path.name)
    return {p.name:sha(p) for p in destination.iterdir()}


def main():
    from experiments.cap_rl import train as trainer
    from analysis import cap_rl_assembly_grammar as grammar, assembly_visual_prompt as visual
    from analysis.train_assembly_preferences import snapshot_parent
    from analysis.cap_rl_assembly_records import audit
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--after-visual-run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--prior-interrupted-run');p.add_argument('--prior-reconstruction-run')
    a=p.parse_args();parent=Path(a.parent).resolve();previous=Path(a.after_visual_run).resolve();out=Path(a.out).resolve()
    previous_meta=read(previous/'metadata.json');selected=read(previous/'selection.json')
    if (previous_meta.get('optimizer_algorithm')!='offline_dpo_v1' or not visual.enabled(previous_meta) or
        selected['completed_optimizer_updates']!=48 or selected['validation_success_rate']>=1. or
        any((previous/n).exists() for n in ('test_comparison.json','raw_test_baseline.json','unconstrained_test_comparison.json'))):
        raise ValueError('Requires completed sub-ceiling visual DPO with untouched held-out outcomes')
    pm=read(parent/'metadata.json');retirement=read(parent/'retirement.json')
    if previous_meta['parent_metadata_sha256']!=sha(parent/'metadata.json'):
        raise ValueError('Visual attempt and online warm start have different parents')
    if retirement['completed_rounds']!=6 or read(parent/'selection.json')['best_step']!=0:
        raise ValueError('Expected the declared GRPO warm start')
    parent_audit=audit(parent,pretest=True)
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'process_start.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),algorithm=ALGORITHM))
    snapshot_parent(parent,out/'parent_run');dump(out/'parent_record_audit.json',parent_audit)
    for name in ('initial_adapter','assembly_source','source','decoding_source','optimization_source'):
        shutil.copytree(parent/name,out/name)
    shutil.copyfile(parent/'assembly_provenance.json',out/'assembly_provenance.json')
    warm=parent/'checkpoints/step-0006';shutil.copytree(warm,out/'warm_start_adapter')
    source_dir=out/'preference_source';source_dir.mkdir()
    names=[*visual.SOURCES,'train_assembly_visual_grpo.py','cap_rl_assembly_records.py']
    for name in names:shutil.copyfile(Path(__file__).with_name(name),source_dir/name)
    previous_receipts=snapshot_attempt(previous,out/'visual_prompt_attempt')
    # Preserve the unsuccessful constant-prompt attempt as separately spent work.
    shutil.copytree(previous/'constant_prompt_attempt',out/'constant_prompt_attempt')
    metadata=dict(pm,optimizer_algorithm=ALGORITHM,prompt_profile=visual.PROFILE,
        steps=8,eval_every=2,checkpoint_every=1,prompts_per_step=4,group_size=4,
        update_epochs=4,microbatch=8,lr=.0001,beta=.005,sampling_temperature=3.,seed=97,
        parent_original_path=str(parent),parent_metadata_sha256=sha(parent/'metadata.json'),
        parent_retirement_sha256=sha(parent/'retirement.json'),
        warm_start_weights_sha256=sha(warm/'adapter_model.safetensors'),
        online_source_sha256={n:sha(source_dir/n) for n in names},
        visual_source_sha256={n:sha(source_dir/n) for n in visual.SOURCES},
        visual_prompt_attempt_sha256=previous_receipts,
        constant_prompt_attempt_sha256=previous_meta['constant_prompt_attempt_sha256'],
        reference_policy='initial',init_adapter=str(out/'initial_adapter'),out=str(out),
        training_scope='On-policy GRPO, one balanced pass over all 32 training layouts, four samples per layout. Initial-RGB text inputs; unchanged strict physical reward.')
    metadata['training_layout_schedule']=schedule(metadata['splits']['train'],metadata['seed'])
    if bool(a.prior_interrupted_run) != bool(a.prior_reconstruction_run):
        raise ValueError('Supply both failed attempt paths')
    if a.prior_interrupted_run:
        attempts={}
        for label,path in (('interrupted',a.prior_interrupted_run),('failed_reconstruction',a.prior_reconstruction_run)):
            source=Path(path);destination=out/'prior_online_attempts'/label;destination.mkdir(parents=True)
            if any((source/n).exists() for n in ('test_comparison.json','selection.json')):
                raise ValueError('Expected interrupted pre-selection attempts')
            for pattern in ('*.json','*.jsonl'):
                for file in source.glob(pattern):shutil.copyfile(file,destination/file.name)
            attempts[label]={file.name:sha(file) for file in destination.iterdir()}
        if read(Path(a.prior_reconstruction_run)/'reconstruction_verification.json')['passes'] is not False:
            raise ValueError('Expected explicitly failed exact reconstruction')
        metadata['prior_online_attempts_sha256']=attempts
        metadata['restart_scope']='Independent rerun from the original warm adapter and seed, not an exact optimizer continuation. Earlier interrupted and failed-reconstruction weights are not loaded; their costs remain separate.'
    state_source=Path(__file__).with_name('assembly_training_state.py')
    shutil.copyfile(state_source,source_dir/state_source.name)
    metadata['training_state_source_sha256']=sha(state_source)
    dump(out/'metadata.json',metadata)
    grammar.install(trainer);trainer.task=trainer.get_task('assembly_full')
    trainer.task.DATASET_PROFILE=metadata['assembly_split']
    trainer.task.PHYSICS_WORKERS=metadata['physics_workers'];trainer.task.PHYSICS_THREADS=metadata['physics_threads']
    trainer.task.configure(metadata['support_profile'],metadata['controller_profile'],metadata['assembly_backend'])
    trainer.task.EXPECTED_PROVENANCE=read(out/'assembly_provenance.json')
    if trainer.task.provenance()!=trainer.task.EXPECTED_PROVENANCE:raise ValueError('Frozen physical environment changed')
    visual.prepare(out,metadata,'development');metadata['observations_manifest_sha256']=sha(out/'observations_manifest.json')
    dump(out/'metadata.json',metadata);dump(out/'observation_preflight.json',visual.audit_observations(out,metadata))
    with visual.installed(out,metadata):
        try:optimize(metadata,out,trainer,grammar)
        finally:trainer.task.shutdown_workers()


def optimize(metadata,out,trainer,grammar):
    import torch
    from transformers import set_seed
    from analysis.assembly_training_state import save as save_training_state
    from analysis.train_assembly_strict import strict_results
    from analysis.cap_rl_assembly_records import metrics
    set_seed(metadata['seed']);torch.set_num_threads(4)
    model,tokenizer=trainer.load_model(metadata['model'],metadata['model_revision'],str(out/'warm_start_adapter'),
        trainable=True,reference_adapter=str(out/'initial_adapter'))
    splits=trainer.task.datasets();layouts_by_key={x.key:x for x in splits['train']}
    def evaluate():
        _,rows=trainer.evaluate(model,tokenizer,splits['validation'],metadata['max_tokens'])
        rows=strict_results(rows);return dict(metrics=metrics(rows),rows=rows)
    with trainer.reference_policy(model):baseline=evaluate()
    dump(out/'validation_baseline.json',baseline)
    shutil.copytree(out/'initial_adapter',out/'best_adapter');tokenizer.save_pretrained(out/'best_adapter')
    best=(baseline['metrics']['success_rate'],baseline['metrics']['mean_reward']);best_step=0
    print(json.dumps(dict(stage='baseline_validation',**baseline['metrics'])),flush=True)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=metadata['lr'])
    started=time.monotonic()
    for step,keys in enumerate(metadata['training_layout_schedule'],1):
        step_started=time.monotonic();layouts=[layouts_by_key[k] for k in keys for _ in range(metadata['group_size'])]
        old=[];reference=[];preparation={}
        @torch.no_grad()
        def prepare(sequences,attention,mask):
            start=time.monotonic();length=mask.shape[1]
            for i in range(0,len(sequences),metadata['microbatch']):
                sl=slice(i,i+metadata['microbatch'])
                old.append(grammar.constrained_logprobs(model,sequences[sl],attention[sl],length,metadata['sampling_temperature']))
                with trainer.reference_policy(model):
                    reference.append(grammar.constrained_logprobs(model,sequences[sl],attention[sl],length,metadata['sampling_temperature']))
            torch.cuda.synchronize();preparation['logprob_s']=time.monotonic()-start
        sequences,attention,mask,texts,results=trainer.rollout(model,tokenizer,layouts,metadata['max_tokens'],True,
            temperature=metadata['sampling_temperature'],prepare=prepare)
        timing=dict(trainer.rollout.last_timing);results=strict_results(results)
        reward=torch.tensor([r['reward'] for r in results],device='cuda');advantage=trainer.advantages(reward,metadata['group_size'])
        old,reference=torch.cat(old),torch.cat(reference);update_start=time.monotonic()
        epoch_metrics=trainer.update_policy(model,optimizer,sequences,attention,mask,old,reference,advantage,
            microbatch=metadata['microbatch'],epochs=metadata['update_epochs'],beta=metadata['beta'],temperature=metadata['sampling_temperature'])
        torch.cuda.synchronize()
        metric=dict(step=step,reward=float(reward.mean()),success_rate=sum(r['success'] for r in results)/len(results),
            active_groups=int((reward.reshape(-1,metadata['group_size']).std(-1)>0).sum()),
            loss=epoch_metrics[-1]['loss'],kl=epoch_metrics[-1]['kl'],grad_norm=epoch_metrics[-1]['grad_norm'],
            epoch_metrics=epoch_metrics,optimizer_steps_total=step*metadata['update_epochs'],
            elapsed_seconds=time.monotonic()-started,step_seconds=time.monotonic()-step_started,
            optimization_seconds=time.monotonic()-update_start,**preparation,**timing,
            sampled_programs=len(results),unique_evidence=len({r['evidence_id'] for r in results if 'evidence_id' in r}),
            cache_hits=sum(bool(r.get('cache_hit')) for r in results),batch_duplicates=sum(bool(r.get('batch_duplicate')) for r in results))
        trainer.append(out/'training.jsonl',[metric]);trainer.append(out/'rollouts.jsonl',[dict(row,step=step) for row in trainer.rows_for(layouts,texts,results)])
        print(json.dumps(metric),flush=True);save_training_state(model,optimizer,out,metric)
        if step%metadata['eval_every']==0:
            report=evaluate();dump(out/f'validation_{step:04d}.json',report)
            print(json.dumps(dict(stage='validation',step=step,**report['metrics'])),flush=True)
            score=(report['metrics']['success_rate'],report['metrics']['mean_reward'])
            if score>best:best,best_step=score,step;trainer.save_policy(model,out/'best_adapter')
            if best==(1.,1.) and best_step>0:break
    trainer.save_policy(model,out/'last_adapter')
    dump(out/'selection.json',dict(best_step=best_step,validation_success_rate=best[0],validation_mean_reward=best[1],
        selection_split='validation',training_seconds=time.monotonic()-started,completed_rollout_rounds=step,
        completed_optimizer_updates=step*metadata['update_epochs'],planned_rollout_rounds=metadata['steps'],
        stopped_at_validation_ceiling=step<metadata['steps'],
        timing_scope='This online phase includes rollout waits, updates and scheduled validation; prior phases and model/input setup are separate.'))


if __name__=='__main__':main()
