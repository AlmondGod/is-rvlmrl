"""Constrained-policy correctness: action space, normalization and gradients."""
from types import SimpleNamespace
import unittest
from analysis import cap_rl_assembly_grammar as module
from experiments.cap_rl.assembly import parse_program


class CharacterTokenizer:
    eos_token_id=255
    def encode(self,text,**kwargs):return list(map(ord,text))
    def __call__(self,text,**kwargs):
        return dict(input_ids=self.encode(text),offset_mapping=[(i,i+1) for i in range(len(text))])


class GrammarTests(unittest.TestCase):
    def setUp(self):self.g=module.AssemblyGrammar(CharacterTokenizer())

    def completion(self,orders):
        values=list(self.g.tokens)
        for group,order in zip(self.g.groups,orders):
            for p,i in zip(group,order):values[p]=self.g.digits[i]
        for p,source in self.g.echoes.items():values[p]=values[source]
        return values

    def test_complete_permutations_and_matching_releases(self):
        values=self.completion([[3,1,0,2],[2,0,3,1],[1,3,2,0]])
        self.g.validate(values)
        calls=parse_program(''.join(map(chr,values[:-1])))
        self.assertEqual(len(calls),22)
        self.assertEqual([i for name,i in calls if name=='position_member'],[3,1,0,2])
        self.assertEqual([i for name,i in calls if name=='release_member'],[3,1,0,2])
        self.assertEqual(len(self.g.choices),12)
        for group in self.g.groups:
            self.assertEqual([len(self.g.allowed(p,values)) for p in group],[4,3,2,1])

    def test_duplicate_missing_and_early_eos_rejected(self):
        values=self.completion([[0,1,2,3]]*3)
        for mutation in ('duplicate','release','eos','short'):
            changed=list(values)
            if mutation=='duplicate':changed[self.g.groups[1][1]]=changed[self.g.groups[1][0]]
            elif mutation=='release':changed[next(iter(self.g.echoes))]=ord('3')
            elif mutation=='eos':changed[0]=255
            else:changed.pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.g.validate(changed)

    def test_conditional_probability_and_gradient_match_masked_sampling(self):
        import torch
        module._GRAMMAR=self.g
        values=self.completion([[3,1,0,2],[2,0,3,1],[1,3,2,0]])
        length=len(values);raw=torch.randn(1,length+1,256,requires_grad=True)
        model=lambda **kwargs:SimpleNamespace(logits=raw)
        seq=torch.tensor([[7,8]+values]);attention=torch.ones_like(seq)
        actual=module.constrained_logprobs(model,seq,attention,length,temperature=2.5)
        expected=[]
        processor=module.GrammarProcessor(self.g,2)
        for p in self.g.choices:
            scores=processor(seq[:,:2+p],raw[:,p])
            expected.append((scores/2.5).log_softmax(-1)[0,values[p]])
        torch.testing.assert_close(actual[0,self.g.choices],torch.stack(expected))
        mask=self.g.decision_mask(torch.ones(1,length))
        self.assertEqual(int(mask.sum()),12)
        (-(actual*mask).sum()/mask.sum()).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        forced=[i for i in range(length) if i not in self.g.choices]
        self.assertEqual(float(raw.grad[:,forced].abs().sum()),0.)
        for p in self.g.choices:
            disallowed=[i for i in range(256) if i not in self.g.allowed(p,values)]
            self.assertEqual(float(raw.grad[:,p,disallowed].abs().sum()),0.)

    def test_rollout_masks_choices_and_restores_generate(self):
        import torch
        tokenizer=CharacterTokenizer();seen=[]
        def generate(**kwargs):
            ids=kwargs['input_ids'];processor=kwargs['logits_processor'][0]
            for p in range(len(self.g.tokens)):
                ids=torch.cat([ids,processor(ids,torch.zeros(len(ids),256)).argmax(-1)[:,None]],dim=1)
            return ids
        model=SimpleNamespace(generate=generate)
        def original(model,tokenizer,layouts,max_tokens,sample,prepare=None,**kwargs):
            seq=model.generate(input_ids=torch.tensor([[7,8]]*len(layouts)))
            attention=torch.ones_like(seq);mask=torch.ones(len(layouts),seq.shape[1]-2)
            if prepare:prepare(seq,attention,mask)
            return seq,attention,mask,['fixture']*len(layouts),[{}]*len(layouts)
        trainer=SimpleNamespace(rollout=original)
        module.install(trainer)
        rows=trainer.rollout(model,tokenizer,[1,2],1024,True,prepare=lambda s,a,m:seen.append(m.sum(-1).tolist()))
        self.assertEqual(seen,[[12.,12.]])
        self.assertEqual(rows[2].sum(-1).tolist(),[12.,12.])
        self.assertIs(model.generate,generate)
        for row in rows[0][:,2:].tolist():self.g.validate(row)

    def test_sampler_produces_distinct_legal_orders(self):
        import torch
        torch.manual_seed(4);ids=torch.tensor([[7,8]]*16)
        processor=module.GrammarProcessor(self.g,2)
        for p in range(len(self.g.tokens)):
            logits=processor(ids,torch.zeros(16,256))
            ids=torch.cat([ids,torch.multinomial(logits.softmax(-1),1)],dim=1)
        completions=ids[:,2:].tolist()
        for c in completions:self.g.validate(c)
        self.assertGreater(len({tuple(c) for c in completions}),10)


if __name__=='__main__':unittest.main()
