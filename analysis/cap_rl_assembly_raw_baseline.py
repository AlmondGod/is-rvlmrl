"""Score raw Qwen on the declared test split only after checkpoint selection."""
import argparse
import json
from pathlib import Path

import torch
from transformers import set_seed
from experiments.cap_rl import train
from analysis.train_assembly_strict import install_execution_filter


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    a=p.parse_args();root=Path(a.run)
    metadata=json.loads((root/'metadata.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    if selection['selection_split']!='validation':
        raise ValueError('Checkpoint selection must be complete before test scoring')
    destination=root/'raw_test_baseline.json'
    if destination.exists():raise FileExistsError(destination)
    task=train.get_task('assembly_full');train.task=task
    task.PHYSICS_WORKERS=metadata['physics_workers']
    task.PHYSICS_THREADS=metadata.get('physics_threads',1)
    task.DATASET_PROFILE=metadata.get('assembly_split','pilot_v1')
    task.configure(metadata['support_profile'],metadata['controller_profile'],metadata['assembly_backend'])
    expected=json.loads((root/'assembly_provenance.json').read_text())
    if task.provenance()!=expected:raise RuntimeError('Source/runtime changed')
    task.EXPECTED_PROVENANCE=expected
    if metadata.get('static_strict_rejection'):
        install_execution_filter(task)
    layouts=task.datasets()['test']
    if [x.key for x in layouts]!=metadata['splits']['test']:raise RuntimeError('Test split changed')
    set_seed(metadata['seed']);torch.set_num_threads(4)
    try:
        model,tokenizer=train.load_model(metadata['model'],metadata['model_revision'])
        metrics,rows=train.evaluate(model,tokenizer,layouts,metadata['max_tokens'])
        for row in rows:
            row.update(dense_reward=row.get('dense_reward',row['reward']),reward=float(row['success']))
        metrics['mean_reward']=metrics['success_rate']
        destination.write_text(json.dumps(dict(metrics=metrics,rows=rows,model_revision=metadata['model_revision'],
            scope='Raw base model, fresh zero-output LoRA initialization, no optimizer steps or checkpoint selection.'),indent=2)+'\n')
        print(json.dumps(metrics),flush=True)
    finally:
        task.shutdown_workers()


if __name__=='__main__':main()
