"""Offline DPO on self-generated, physically certified assembly preferences.

This phase follows an explicitly retired GRPO pilot. It does not call sampled
programs new environment rollouts: every optimizer batch reuses certified
training evidence. Only scheduled greedy validation runs new physics as needed.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_loss(policy, reference, beta):
    import torch.nn.functional as F
    margin = (policy[:,0]-reference[:,0])-(policy[:,1]-reference[:,1])
    return F.softplus(-beta*margin).mean(), margin


def sample_pairs(data, rng, count):
    groups=defaultdict(list)
    for i,pair in enumerate(data['pairs']):groups[pair['layout_id']].append(i)
    layouts=sorted(groups)
    if not layouts:raise ValueError('No within-layout strict success/failure preferences')
    return [rng.choice(groups[rng.choice(layouts)]) for _ in range(count)]


def program_inventory(data, first_layout, visual):
    """Identical code under different sensor contexts needs distinct log-probs."""
    return list(dict.fromkeys((pair['layout_id'] if visual else first_layout, pair[role])
                for pair in data['pairs'] for role in ('chosen','rejected')))


def snapshot_parent(parent, destination):
    for p in parent.rglob('*'):
        rel=p.relative_to(parent)
        if not p.is_file() or '__pycache__' in rel.parts:continue
        if 'checkpoints' in rel.parts and p.name!='checkpoint.json':continue
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)


def main():
    from experiments.cap_rl import train as trainer
    from analysis import cap_rl_assembly_grammar as grammar
    from analysis import train_assembly_strict as strict
    from analysis.assembly_preference_data import build
    from analysis.cap_rl_assembly_records import audit, metrics

    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=48);p.add_argument('--eval-every',type=int,default=8)
    p.add_argument('--pairs-per-step',type=int,default=4);p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--beta',type=float,default=.1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--visual-observations',action='store_true')
    p.add_argument('--after-constant-run',help='Required completed unsuccessful constant-prompt phase for the visual contingency')
    args=p.parse_args()
    if (args.steps,args.eval_every,args.pairs_per_step,args.lr,args.beta,args.seed)!=(48,8,4,.0001,.1,42):
        raise ValueError('This entry point implements the recorded initial preference-replay protocol')
    parent,out=Path(args.parent).resolve(),Path(args.out).resolve()
    prior_attempt=None
    if args.visual_observations:
        if not args.after_constant_run:raise ValueError('Visual contingency requires the completed constant-prompt phase')
        prior_attempt=Path(args.after_constant_run).resolve()
        prior_meta=read(prior_attempt/'metadata.json');prior_selection=read(prior_attempt/'selection.json')
        if (prior_meta.get('optimizer_algorithm')!='offline_dpo_v1' or prior_meta.get('prompt_profile','constant')!='constant' or
            prior_selection['completed_optimizer_updates']!=48 or prior_selection['validation_success_rate']>=1. or
            any((prior_attempt/n).exists() for n in ('test_comparison.json','raw_test_baseline.json','unconstrained_test_comparison.json'))):
            raise ValueError('Visual contingency must follow a completed sub-ceiling phase with untouched held-out data')
    elif args.after_constant_run:
        raise ValueError('--after-constant-run is only for the declared visual contingency')
    pm=read(parent/'metadata.json');retirement=read(parent/'retirement.json')
    if retirement['completed_rounds']!=6 or read(parent/'selection.json')['best_step']!=0:
        raise ValueError('Parent must meet the declared six-round unchanged-policy contingency')
    if any((parent/n).exists() for n in ('test_comparison.json','raw_test_baseline.json','unconstrained_test_comparison.json')):
        raise ValueError('No held-out evaluation may precede preference training')
    parent_audit=audit(parent,pretest=True)
    data=build(parent)
    if not data['pairs']:raise ValueError('No certified preferences')
    out.mkdir(parents=True,exist_ok=False)
    trainer.dump(out/'process_start.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),algorithm='offline_dpo_v1'))
    snapshot_parent(parent,out/'parent_run')
    trainer.dump(out/'parent_record_audit.json',parent_audit)
    trainer.dump(out/'preference_data.json',data)
    for name in ('initial_adapter','assembly_source','source','decoding_source','optimization_source'):
        shutil.copytree(parent/name,out/name)
    shutil.copyfile(parent/'assembly_provenance.json',out/'assembly_provenance.json')
    tools=out/'preference_source';tools.mkdir()
    names=['train_assembly_preferences.py','assembly_preference_data.py','cap_rl_assembly_records.py','retire_assembly_plateau.py']
    if args.visual_observations:
        from analysis.assembly_visual_prompt import SOURCES
        names+=list(SOURCES)
    for name in names:
        source=Path(__file__).with_name(name)
        shutil.copyfile(source,tools/name)
    warm=parent/'checkpoints/step-0006'
    shutil.copytree(warm,out/'warm_start_adapter')
    metadata=dict(pm,optimizer_algorithm='offline_dpo_v1',steps=args.steps,eval_every=args.eval_every,
        checkpoint_every=args.eval_every,update_epochs=1,prompts_per_step=args.pairs_per_step,group_size=2,
        microbatch=2,lr=args.lr,dpo_beta=args.beta,seed=args.seed,sampling_temperature=1.,
        parent_original_path=str(parent),parent_metadata_sha256=sha(parent/'metadata.json'),
        parent_retirement_sha256=sha(parent/'retirement.json'),preference_data_sha256=sha(out/'preference_data.json'),
        warm_start_weights_sha256=sha(warm/'adapter_model.safetensors'),
        preference_source_sha256={name:sha(tools/name) for name in names},
        training_scope='Offline preference updates; no new sampled physical rollouts. Scheduled greedy validation retains full physics.',
        reference_policy='initial',init_adapter=str(out/'initial_adapter'),out=str(out))
    trainer.dump(out/'metadata.json',metadata)
    grammar.install(trainer)
    trainer.task=trainer.get_task('assembly_full')
    trainer.task.DATASET_PROFILE=metadata['assembly_split']
    trainer.task.PHYSICS_WORKERS=metadata['physics_workers'];trainer.task.PHYSICS_THREADS=metadata['physics_threads']
    trainer.task.configure(metadata['support_profile'],metadata['controller_profile'],metadata['assembly_backend'])
    trainer.task.EXPECTED_PROVENANCE=read(out/'assembly_provenance.json')
    if trainer.task.provenance()!=trainer.task.EXPECTED_PROVENANCE:
        raise ValueError('Use the parent frozen physical source/runtime')
    if args.visual_observations:
        from analysis import assembly_visual_prompt as visual
        metadata.update(prompt_profile=visual.PROFILE,
            visual_source_sha256={name:sha(tools/name) for name in visual.SOURCES})
        visual.prepare(out,metadata,'development')
        metadata['observations_manifest_sha256']=sha(out/'observations_manifest.json')
        data=visual.augment_preferences(data,out,metadata)
        trainer.dump(out/'preference_data.json',data)
        metadata['preference_data_sha256']=sha(out/'preference_data.json')
        context=out/'constant_prompt_attempt';context.mkdir()
        for name in ['metadata.json','selection.json','training.jsonl','rollouts.jsonl','validation_baseline.json','process_start.json']:
            shutil.copyfile(prior_attempt/name,context/name)
        for path in prior_attempt.glob('validation_[0-9]*.json'):shutil.copyfile(path,context/path.name)
        metadata['constant_prompt_attempt_sha256']={path.name:sha(path) for path in context.iterdir()}
        metadata['training_scope']='Offline DPO on own historical, certified training programs with scene-matched initial RGB estimates. No new training physics; full scheduled validation.'
        trainer.dump(out/'metadata.json',metadata)
        trainer.dump(out/'observation_preflight.json',visual.audit_observations(out,metadata))
    from contextlib import nullcontext
    prompt_context=visual.installed(out,metadata) if args.visual_observations else nullcontext()
    # This context is also closed if loading, baseline evaluation or setup fails.
    with prompt_context:
        optimize(args,metadata,data,out,trainer,grammar,strict,metrics)


def optimize(args,metadata,data,out,trainer,grammar,strict,metrics):
    import torch
    from transformers import set_seed
    from experiments.cap_rl.checkpoints import save_step_adapter
    set_seed(args.seed);torch.set_num_threads(4)
    model,tokenizer=trainer.load_model(metadata['model'],metadata['model_revision'],str(out/'warm_start_adapter'),
        trainable=True,reference_adapter=str(out/'initial_adapter'))
    splits=trainer.task.datasets()

    def evaluate():
        _,rows=trainer.evaluate(model,tokenizer,splits['validation'],metadata['max_tokens'])
        rows=strict.strict_results(rows)
        return dict(metrics=metrics(rows),rows=rows)

    with trainer.reference_policy(model):baseline=evaluate()
    trainer.dump(out/'validation_baseline.json',baseline)
    shutil.copytree(out/'initial_adapter',out/'best_adapter');tokenizer.save_pretrained(out/'best_adapter')
    best=(baseline['metrics']['success_rate'],baseline['metrics']['mean_reward']);best_step=0
    print(json.dumps(dict(stage='baseline_validation',**baseline['metrics'])),flush=True)
    grammar._GRAMMAR=grammar.AssemblyGrammar(tokenizer)
    source_rows=[json.loads(line) for line in (out/'parent_run/rollouts.jsonl').read_text().splitlines()]
    use_visual=metadata.get('prompt_profile')=='initial_rgb_member_estimates_v1'
    key=lambda pair,role:(pair['layout_id'] if use_visual else splits['train'][0].key,pair[role])
    all_keys=program_inventory(data,splits['train'][0].key,use_visual)
    indices={item:i for i,item in enumerate(all_keys)}
    completion_ids=[]
    for layout_id,text in all_keys:
        ids=tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id]
        grammar._GRAMMAR.validate(ids);completion_ids.append(ids)
    layouts={layout.key:layout for layout in splits['train']}
    inputs=trainer.encode(tokenizer,[layouts[layout_id] for layout_id,text in all_keys])
    completion=torch.tensor(completion_ids,device='cuda')
    sequences=torch.cat([inputs.input_ids,completion],dim=1)
    attention=torch.cat([inputs.attention_mask,torch.ones_like(completion)],dim=1)
    length=completion.shape[1]
    with torch.no_grad(),trainer.reference_policy(model):
        reference=torch.cat([grammar.constrained_logprobs(model,sequences[i:i+2],attention[i:i+2],length,1.).sum(-1)
                             for i in range(0,len(sequences),2)]).detach()
    if reference.requires_grad:raise RuntimeError('Reference must be frozen')
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
    rng=random.Random(args.seed);started=time.monotonic()
    try:
        for step in range(1,args.steps+1):
            update_start=time.monotonic();pair_ids=sample_pairs(data,rng,args.pairs_per_step)
            optimizer.zero_grad(set_to_none=True);loss_total=0.;correct=0;replayed=[]
            for pair_id in pair_ids:
                pair=data['pairs'][pair_id]
                take=torch.tensor([indices[key(pair,'chosen')],indices[key(pair,'rejected')]],device='cuda')
                logp=grammar.constrained_logprobs(model,sequences[take],attention[take],length,1.).sum(-1)
                loss,margin=pair_loss(logp[None,:],reference[take][None,:],args.beta)
                (loss/args.pairs_per_step).backward()
                loss_total+=float(loss.detach())/args.pairs_per_step
                correct+=int((margin.detach()>0).sum())
                for role in ('chosen','rejected'):
                    row=source_rows[pair[role+'_row']]
                    replayed.append(dict(row,prompt=pair['prompt'],step=step,preference_pair_id=pair_id,preference_role=role,
                                         parent_source_row=pair[role+'_row'],offline_replay=True))
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            if not all(x==x and abs(x)!=float('inf') for x in (norm,loss_total)):
                raise RuntimeError('Nonfinite preference update')
            optimizer.step();torch.cuda.synchronize()
            metric=dict(step=step,optimizer_steps_total=step,loss=loss_total,grad_norm=norm,
                preference_accuracy=correct/args.pairs_per_step,pair_indices=pair_ids,
                elapsed_seconds=time.monotonic()-started,optimization_seconds=time.monotonic()-update_start,
                new_training_physics_episodes=0,replayed_programs=len(replayed))
            trainer.append(out/'training.jsonl',[metric]);trainer.append(out/'rollouts.jsonl',replayed)
            print(json.dumps(metric),flush=True)
            if step%args.eval_every==0:
                save_step_adapter(model,out,metric)
                report=evaluate();trainer.dump(out/f'validation_{step:04d}.json',report)
                print(json.dumps(dict(stage='validation',step=step,**report['metrics'])),flush=True)
                score=report['metrics']['success_rate'],report['metrics']['mean_reward']
                if score>best:best,best_step=score,step;trainer.save_policy(model,out/'best_adapter')
                if best==(1.,1.):break
        trainer.save_policy(model,out/'last_adapter')
        trainer.dump(out/'selection.json',dict(best_step=best_step,validation_success_rate=best[0],
            validation_mean_reward=best[1],selection_split='validation',training_seconds=time.monotonic()-started,
            completed_optimizer_updates=step,planned_optimizer_updates=args.steps,
            stopped_at_validation_ceiling=best==(1.,1.) and step<args.steps,
            timing_scope='Preference updates and scheduled validation; parent GRPO and this phase model setup/baseline costs are separate.'))
    finally:trainer.task.shutdown_workers()


if __name__=='__main__':main()
