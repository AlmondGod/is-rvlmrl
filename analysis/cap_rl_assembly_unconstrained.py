"""Diagnose free-form code generation from frozen initial/selected RL weights.

Training uses the declared constrained action space. This secondary evaluation
removes that decoding restriction without repairing programs. Validation is a
diagnostic; held-out evaluation follows the completed primary comparison and
must not be used for checkpoint/decoder selection or further tuning.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
from transformers import set_seed
from experiments.cap_rl import train
from analysis.train_assembly_strict import install_execution_filter, strict_results


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--split',choices=('validation','test'),required=True)
    args=p.parse_args();root=Path(args.run)
    metadata=json.loads((root/'metadata.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    if metadata.get('policy_decoding')!='assembly_permutations_v1' or selection['selection_split']!='validation':
        raise ValueError('This comparison requires completed constrained-policy validation selection')
    if args.split=='test' and not (root/'test_comparison.json').exists():
        raise ValueError('Complete the primary held-out evaluation before this secondary comparison')
    destination=root/f'unconstrained_{args.split}_comparison.json'
    if destination.exists():raise FileExistsError(destination)
    task=train.get_task('assembly_full');train.task=task
    task.PHYSICS_WORKERS=metadata['physics_workers'];task.PHYSICS_THREADS=metadata.get('physics_threads',1)
    task.DATASET_PROFILE=metadata['assembly_split']
    task.configure(metadata['support_profile'],metadata['controller_profile'],metadata['assembly_backend'])
    expected=json.loads((root/'assembly_provenance.json').read_text())
    if task.provenance()!=expected:raise RuntimeError('Physical source/runtime changed')
    task.EXPECTED_PROVENANCE=expected;install_execution_filter(task)
    layouts=task.datasets()[args.split]
    if [x.key for x in layouts]!=metadata['splits'][args.split]:raise ValueError('Split mismatch')
    records={};metrics={}
    set_seed(metadata['seed']);torch.set_num_threads(4)
    try:
        for label,adapter in [('before',root/'initial_adapter'),('after',root/'best_adapter')]:
            model,tokenizer=train.load_model(metadata['model'],metadata['model_revision'],str(adapter))
            _,rows=train.evaluate(model,tokenizer,layouts,metadata['max_tokens'])
            rows=strict_results(rows)
            records[label]=rows
            metrics[label]=dict(n=len(rows),successes=sum(r['success'] for r in rows),
                success_rate=sum(r['success'] for r in rows)/len(rows),
                mean_reward=sum(r['reward'] for r in rows)/len(rows),valid_rate=sum(r['valid'] for r in rows)/len(rows))
            print(json.dumps(dict(policy=label,split=args.split,**metrics[label])),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
        source=root/'evaluation_source'/Path(__file__).name;source.parent.mkdir(exist_ok=True)
        data=Path(__file__).read_bytes()
        if source.exists() and source.read_bytes()!=data:raise ValueError('Evaluation source changed')
        source.write_bytes(data)
        wins=sum(not a['success'] and b['success'] for a,b in zip(records['before'],records['after']))
        losses=sum(a['success'] and not b['success'] for a,b in zip(records['before'],records['after']))
        report=dict(format='assembly_unconstrained_transfer_v1',split=args.split,
            evaluation_decoding='unrestricted_program',selection_role='diagnostic_only',
            model_revision=metadata['model_revision'],metadata_sha256=sha(root/'metadata.json'),
            selection_sha256=sha(root/'selection.json'),evaluator_source_sha256=sha(source),
            initial_weights_sha256=sha(root/'initial_adapter/adapter_model.safetensors'),
            selected_weights_sha256=sha(root/'best_adapter/adapter_model.safetensors'),
            before=records['before'],after=records['after'],baseline=metrics['before'],trained=metrics['after'],
            paired_wins=wins,paired_losses=losses,delta_success_rate=(wins-losses)/len(layouts),
            scope='Frozen-weight secondary greedy evaluation without decoding grammar or program repair. No optimization or checkpoint selection.')
        destination.write_text(json.dumps(report,indent=2)+'\n')
    finally:task.shutdown_workers()


if __name__=='__main__':main()
