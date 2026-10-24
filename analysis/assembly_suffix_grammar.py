"""Experimental conditional panel-nailing curriculum, not a full-plan policy.

A separate frozen prefix policy must supply each complete, legal source plan.
This sampler preserves that plan through sheet placement/release and learns only
the four panel-nailing indices (24 orders). Every generated program still runs
the complete physical assembly and flip. Callers must disclose this restricted
action space and bind prefix plans to their actual frozen-model provenance.
"""
from analysis import cap_rl_assembly_grammar as full


class FixedPrefixProcessor:
    def __init__(self, grammar, prompt_length, programs, tokenizer):
        self.grammar=grammar;self.prompt_length=prompt_length
        self.cutoff=grammar.groups[2][0]
        self.prefixes=[]
        for program in programs:
            tokens=tokenizer.encode(program,add_special_tokens=False)+[tokenizer.eos_token_id]
            grammar.validate(tokens)
            self.prefixes.append(tokens[:self.cutoff])

    def __call__(self,input_ids,scores):
        import torch
        if len(input_ids)!=len(self.prefixes):
            raise ValueError('Prefix batch changed; beam expansion is unsupported')
        position=input_ids.shape[1]-self.prompt_length
        if position<0:raise ValueError('Input shorter than prompt')
        if position>=self.cutoff:return scores
        completions=input_ids[:,self.prompt_length:].tolist()
        allowed=torch.zeros_like(scores,dtype=torch.bool)
        for i,(prefix,completion) in enumerate(zip(self.prefixes,completions)):
            if completion!=prefix[:position]:
                raise ValueError('Generated code changed the frozen prefix')
            token=prefix[position]
            if token not in self.grammar.allowed(position,completion):
                raise ValueError('Frozen prefix violates the assembly grammar')
            allowed[i,token]=True
        return scores.masked_fill(~allowed,float('-inf'))


def generation_kwargs(tokenizer,prompt_length,max_tokens,prefix_programs):
    if not prefix_programs:raise ValueError('Frozen prefix programs are required')
    grammar=full.AssemblyGrammar(tokenizer)
    if max_tokens<len(grammar.tokens):raise ValueError('Token budget cannot fit complete assembly code')
    grammar.choices=list(grammar.groups[2])
    fixed=FixedPrefixProcessor(grammar,prompt_length,prefix_programs,tokenizer)
    full._GRAMMAR=grammar
    return dict(logits_processor=[full.GrammarProcessor(grammar,prompt_length),fixed])


def validate(completions,prefix_programs,tokenizer):
    grammar=full._GRAMMAR
    if grammar is None or grammar.choices!=grammar.groups[2]:
        raise ValueError('Panel-nailing grammar is not installed')
    fixed=FixedPrefixProcessor(grammar,0,prefix_programs,tokenizer)
    if len(completions)!=len(fixed.prefixes):raise ValueError('Missing program/prefix')
    for tokens,prefix in zip(completions,fixed.prefixes):
        grammar.validate(tokens)
        if list(tokens[:fixed.cutoff])!=prefix:
            raise ValueError('Completion differs from its frozen prefix policy')
