"""Constrained program GRPO over three permutations of the full assembly plan.

This explicitly narrows the policy's action space to complete legal skill plans;
no sampled program is repaired and no successful ordering is supplied. Only the
12 order-choice tokens have policy probability/gradient. Forced syntax and
prerequisite calls have probability one. The unchanged full physical evaluator
scores every candidate, including all geometry/fastener/flip/hold requirements.
"""
import hashlib
import json
from pathlib import Path
import re

_GRAMMAR = None


class AssemblyGrammar:
    def __init__(self, tokenizer):
        lines = [line for i in range(4) for line in (f'position_member({i})', f'release_member({i})')]
        lines += ['park_grippers()'] + [f'nail_corner({i})' for i in range(4)]
        lines += ['feed_panel()', 'position_panel()', 'release_panel()']
        lines += [f'nail_panel({i})' for i in range(4)] + ['flip_structure()', 'finish()']
        source = '\n'.join(lines)
        encoded = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
        self.tokens = list(encoded['input_ids']) + [tokenizer.eos_token_id]
        offsets = [tuple(x) for x in encoded['offset_mapping']]
        self.digits = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(4)]
        if any(len(x)!=1 for x in self.digits):raise ValueError('Grammar requires single-token literal indices')
        self.digits = [x[0] for x in self.digits]
        if len(set(self.digits))!=4:raise ValueError('Index tokens must be distinct')
        self.groups = [[], [], []]
        self.echoes = {}
        self.choice_group = {}
        for match in re.finditer(r'(position_member|release_member|nail_corner|nail_panel)\(([0-3])\)', source):
            name=match.group(1);span=match.span(2)
            if offsets.count(span)!=1:raise ValueError('Index is not an independent program token')
            position=offsets.index(span)
            if name=='release_member':self.echoes[position]=self.groups[0][-1]
            else:
                group={'position_member':0,'nail_corner':1,'nail_panel':2}[name]
                self.groups[group].append(position);self.choice_group[position]=group
        if [len(g) for g in self.groups]!=[4,4,4]:raise ValueError('Incomplete grammar')
        self.choices=sorted(self.choice_group)

    def allowed(self, position, completion):
        if position in self.choice_group:
            prior=self.groups[self.choice_group[position]]
            used={int(completion[p]) for p in prior if p<position}
            return [d for d in self.digits if d not in used]
        if position in self.echoes:return [int(completion[self.echoes[position]])]
        if position>=len(self.tokens):raise ValueError('Generation continued beyond required EOS')
        return [self.tokens[position]]

    def validate(self, completion):
        if len(completion)!=len(self.tokens):raise ValueError('Incomplete constrained program')
        for i,token in enumerate(completion):
            if int(token) not in self.allowed(i,completion):raise ValueError('Program violates declared decoding grammar')

    def decision_mask(self, mask):
        if mask.shape[1]!=len(self.tokens):raise ValueError('Unexpected completion length')
        result=mask.new_zeros(mask.shape)
        result[:,self.choices]=mask[:,self.choices]
        return result


class GrammarProcessor:
    def __init__(self, grammar, prompt_length):
        self.grammar,self.prompt_length=grammar,prompt_length

    def __call__(self, input_ids, scores):
        import torch
        position=input_ids.shape[1]-self.prompt_length
        allowed=torch.zeros_like(scores,dtype=torch.bool)
        completions=input_ids[:,self.prompt_length:].tolist()
        for i,completion in enumerate(completions):
            allowed[i,self.grammar.allowed(position,completion)]=True
        return scores.masked_fill(~allowed,float('-inf'))


def generation_kwargs(tokenizer, prompt_length, max_tokens):
    global _GRAMMAR
    grammar=AssemblyGrammar(tokenizer)
    if max_tokens<len(grammar.tokens):raise ValueError('Token budget cannot fit the declared grammar')
    _GRAMMAR=grammar
    return dict(logits_processor=[GrammarProcessor(grammar,prompt_length)])


def constrained_logprobs(model, sequences, attention, completion_length, temperature=1.):
    import torch
    grammar=_GRAMMAR
    if grammar is None or completion_length!=len(grammar.tokens):raise ValueError('Missing/mismatched grammar')
    position_ids=attention.long().cumsum(-1)-1
    position_ids.masked_fill_(attention==0,1)
    # Normalize over the actually available index tokens, matching sampling.
    # Qwen supports selecting hidden-state positions before its vocabulary head.
    # Only 12 distributions are needed; avoid materializing logits for syntax.
    positions=torch.tensor(grammar.choices,device=sequences.device)+sequences.shape[1]-completion_length-1
    logits=model(input_ids=sequences,attention_mask=attention,position_ids=position_ids,
                 use_cache=False,logits_to_keep=positions).logits
    labels=sequences[:,-completion_length:]
    completions=labels.tolist()
    for completion in completions:grammar.validate(completion)
    rows=[]
    for choice,position in enumerate(grammar.choices):
        choices=logits[:,choice,grammar.digits].float()/temperature
        allowed=torch.tensor([[d in grammar.allowed(position,c) for d in grammar.digits]
                              for c in completions],device=choices.device,dtype=torch.bool)
        normalized=choices.masked_fill(~allowed,float('-inf')).log_softmax(-1)
        targets=torch.tensor([grammar.digits.index(int(c[position])) for c in completions],device=choices.device)
        rows.append(normalized.gather(1,targets[:,None]).squeeze(1))
    result=logits.new_zeros((len(sequences),completion_length),dtype=torch.float32)
    result[:,grammar.choices]=torch.stack(rows,dim=1)
    return result


def verify_source(directory,metadata):
    source=Path(__file__).read_bytes()
    if metadata.get('policy_decoding')!='assembly_permutations_v1':raise ValueError('Wrong decoding protocol')
    if hashlib.sha256(source).hexdigest()!=metadata['decoding_source_sha256']:
        raise ValueError('Use the exact recorded constrained sampler')
    if (Path(directory)/'decoding_source'/Path(__file__).name).read_bytes()!=source:
        raise ValueError('Recorded constrained sampler changed')


def install(trainer):
    if getattr(trainer.rollout,'_assembly_grammar',False):return
    original_rollout=trainer.rollout
    def rollout(model,tokenizer,layouts,max_tokens,sample,**kwargs):
        global _GRAMMAR
        # The original rollout preserves generation timing, async simulation,
        # attention masks and truncation handling. Only decoding and RL mask vary.
        original_generate=model.generate
        prepare=kwargs.get('prepare')
        if prepare is not None:
            kwargs['prepare']=lambda seq,attention,mask:prepare(seq,attention,_GRAMMAR.decision_mask(mask))
        def generate(*args,**generation):
            generation.update(generation_kwargs(tokenizer,generation['input_ids'].shape[1],max_tokens))
            return original_generate(*args,**generation)
        model.generate=generate
        try:
            seq,attention,mask,texts,results=original_rollout(model,tokenizer,layouts,max_tokens,sample,**kwargs)
        finally:model.generate=original_generate
        rollout.last_timing=dict(getattr(original_rollout,'last_timing',{}))
        for completion in seq[:,-mask.shape[1]:].tolist():_GRAMMAR.validate(completion)
        return seq,attention,_GRAMMAR.decision_mask(mask),texts,results
    rollout._assembly_grammar=True
    trainer.rollout=rollout
    trainer.token_logprobs=constrained_logprobs


def main():
    from experiments.cap_rl import train as trainer
    from analysis import train_assembly_strict as strict
    install(trainer)
    source=Path(__file__).read_bytes();digest=hashlib.sha256(source).hexdigest()
    original_train,original_test,original_dump=trainer.train,trainer.test,trainer.dump
    def train(args):
        args.policy_decoding='assembly_permutations_v1'
        args.decoding_source_sha256=digest
        args.policy_choice_tokens=12
        args.policy_action_space=24**3
        original_train(args)
    def test(args):
        metadata=json.loads((Path(args.out)/'metadata.json').read_text())
        verify_source(args.out,metadata)
        original_test(args)
    def dump(path,value):
        original_dump(path,value)
        if Path(path).name=='metadata.json':
            directory=Path(path).parent/'decoding_source';directory.mkdir(exist_ok=False)
            (directory/Path(__file__).name).write_bytes(source)
    trainer.train,trainer.test,trainer.dump=train,test,dump
    strict.main()


if __name__=='__main__':main()
