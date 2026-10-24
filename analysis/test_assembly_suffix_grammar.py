from types import SimpleNamespace
import unittest

import torch

from analysis import cap_rl_assembly_grammar as full,assembly_suffix_grammar as suffix
from analysis.test_assembly_grammar import CharacterTokenizer


def program(grammar,orders):
    values=list(grammar.tokens)
    for group,order in zip(grammar.groups,orders):
        for pos,digit in zip(group,order):values[pos]=grammar.digits[digit]
    for pos,source in grammar.echoes.items():values[pos]=values[source]
    return ''.join(map(chr,values[:-1]))


class SuffixGrammarTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer=CharacterTokenizer();self.original=full.AssemblyGrammar(self.tokenizer)
        self.programs=[program(self.original,orders) for orders in (
            [[0,1,2,3],[0,1,2,3],[0,1,2,3]],
            [[3,1,0,2],[2,0,3,1],[1,3,2,0]])]

    def test_preserves_distinct_prefixes_and_samples_legal_suffixes(self):
        torch.manual_seed(71)
        programs=self.programs*8
        processors=suffix.generation_kwargs(self.tokenizer,2,1024,programs)['logits_processor']
        ids=torch.tensor([[7,8]]*16)
        for _ in range(len(full._GRAMMAR.tokens)):
            scores=torch.zeros(16,256)
            for processor in processors:scores=processor(ids,scores)
            ids=torch.cat([ids,torch.multinomial(scores.softmax(-1),1)],dim=1)
        completions=ids[:,2:].tolist();suffix.validate(completions,programs,self.tokenizer)
        self.assertGreater(len({tuple(c[full._GRAMMAR.groups[2][0]:]) for c in completions}),6)
        self.assertEqual(full._GRAMMAR.decision_mask(torch.ones(16,len(completions[0]))).sum(-1).tolist(),[4.]*16)

    def test_logprob_and_gradient_only_include_free_suffix_choices(self):
        processors=suffix.generation_kwargs(self.tokenizer,2,1024,[self.programs[1]])['logits_processor']
        grammar=full._GRAMMAR;values=self.tokenizer.encode(self.programs[1])+[255]
        raw=torch.randn(1,len(values)+1,256,requires_grad=True)
        model=lambda **kwargs:SimpleNamespace(logits=raw[:,kwargs['logits_to_keep']-1])
        sequence=torch.tensor([[7,8]+values]);attention=torch.ones_like(sequence)
        actual=full.constrained_logprobs(model,sequence,attention,len(values),2.5)
        expected=[]
        for pos in grammar.choices:
            scores=raw[:,pos]
            for processor in processors:scores=processor(sequence[:,:2+pos],scores)
            expected.append((scores/2.5).log_softmax(-1)[0,values[pos]])
        torch.testing.assert_close(actual[0,grammar.choices],torch.stack(expected))
        actual.sum().backward()
        forced=[pos for pos in range(len(values)) if pos not in grammar.choices]
        self.assertEqual(float(raw.grad[:,forced].abs().sum()),0.)
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_changed_prefix_rejected_even_when_full_program_is_legal(self):
        suffix.generation_kwargs(self.tokenizer,2,1024,[self.programs[0]])
        with self.assertRaisesRegex(ValueError,'frozen prefix'):
            suffix.validate([self.tokenizer.encode(self.programs[1])+[255]],[self.programs[0]],self.tokenizer)


if __name__=='__main__':unittest.main()
