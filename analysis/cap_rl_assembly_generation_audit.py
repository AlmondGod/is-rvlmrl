"""Regenerate both held-out greedy programs from the frozen base/selected weights.

Does not invoke physics, optimize weights, or select checkpoints.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import set_seed
from experiments.cap_rl import train


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();root=Path(args.run)
    metadata=json.loads((root/'metadata.json').read_text())
    report=json.loads((root/'test_comparison.json').read_text())
    constrained=bool(metadata.get('policy_decoding'))
    if constrained:
        from analysis.cap_rl_assembly_grammar import verify_source,generation_kwargs
        verify_source(root,metadata)
    train.task=train.get_task('assembly_full')
    train.task.configure(metadata['support_profile'],metadata['controller_profile'],
                         **({'assembly_backend': metadata['assembly_backend']}
                            if 'assembly_backend' in metadata else {}))
    assert train.task.provenance()==json.loads((root/'assembly_provenance.json').read_text())
    initial = root/'initial_adapter' if metadata.get('init_adapter') else None
    if initial is not None:
        for name,digest in metadata['initial_adapter_sha256'].items():
            assert hashlib.sha256((initial/name).read_bytes()).hexdigest()==digest
    train.task.DATASET_PROFILE=metadata.get('assembly_split','pilot_v1')
    layouts=train.task.datasets()['test']
    set_seed(metadata['seed']);torch.set_num_threads(4)
    hashes={}
    policies=[('before',str(initial) if initial is not None else None),('after',str(root/'best_adapter'))]
    if (root/'raw_test_baseline.json').exists():
        report['raw']=json.loads((root/'raw_test_baseline.json').read_text())['rows']
        policies.append(('raw',None))
    if (root/'unconstrained_test_comparison.json').exists():
        transfer=json.loads((root/'unconstrained_test_comparison.json').read_text())
        for label,adapter in [('before',str(initial)),('after',str(root/'best_adapter'))]:
            report['unconstrained_'+label]=transfer[label]
            policies.append(('unconstrained_'+label,adapter))
    for label,adapter in policies:
        model,tokenizer=train.load_model(metadata['model'],metadata['model_revision'],adapter)
        texts=[];truncated=[]
        for start in range(0,len(layouts),16):
            inputs=train.encode(tokenizer,layouts[start:start+16])
            kwargs=(generation_kwargs(tokenizer,inputs.input_ids.shape[1],metadata['max_tokens'])
                    if constrained and label in ('before','after') else {})
            with torch.no_grad():
                generated=model.generate(**inputs,max_new_tokens=metadata['max_tokens'],do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,**kwargs)
            completions=generated[:,inputs.input_ids.shape[1]:]
            texts.extend(tokenizer.batch_decode(completions,skip_special_tokens=True))
            truncated.extend((~completions.eq(tokenizer.eos_token_id).any(-1)).tolist())
        assert len(texts)==len(report[label])==len(layouts)
        for i,(text,row,layout) in enumerate(zip(texts,report[label],layouts)):
            assert row['layout_id']==layout.key and row['prompt']==train.task.prompt(layout)
            assert text==row['completion'], (label,layout.key,'generated program differs')
            assert row['truncated']==truncated[i]
        hashes[label]=[hashlib.sha256(t.encode()).hexdigest() for t in texts]
        del model,generated,completions,inputs;gc.collect();torch.cuda.empty_cache()
    result=dict(passes=True,programs_regenerated=len(policies)*len(layouts),greedy=True,model_revision=metadata['model_revision'],
        evaluation_inputs_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('metadata.json','selection.json','test_comparison.json',
                         'raw_test_baseline.json','unconstrained_test_comparison.json') if (root/name).exists()},
        selected_adapter_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'best_adapter').iterdir() if p.is_file()},
        program_sha256=hashes,initial_adapter=initial is not None,
        scope='Exact regeneration from the recorded initial policy, validation-selected adapter and raw model when present. Includes declared unrestricted transfer comparisons. No optimization or physics execution.')
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
