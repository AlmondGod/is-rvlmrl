"""Check pipeline ordering and sampling probabilities without a GPU or simulation."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.cap_rl import train, task_assembly


class ParallelRollouts(unittest.TestCase):
    def test_reused_rollouts_keep_old_probabilities_and_microbatch_weighting(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.scores=torch.nn.Parameter(torch.tensor([.1,.2,-.1]))
            def forward(self,input_ids,**kwargs):
                return SimpleNamespace(logits=self.scores.expand(input_ids.shape[0],2,3))
        tokens=torch.tensor([[0,1],[0,2]]); attention=torch.ones_like(tokens)
        mask=torch.ones(2,1); advantage=torch.tensor([1.,-1.]); weights=[]
        for microbatch in (1,2):
            model=Model(); optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            old=train.token_logprobs(model,tokens,attention,1,1.2).detach().clone()
            original=old.clone()
            with patch.object(optimizer,'step',wraps=optimizer.step) as step:
                metrics=train.update_policy(model,optimizer,tokens,attention,mask,old,old,advantage,
                    microbatch=microbatch,epochs=3,beta=.02,temperature=1.2)
                self.assertEqual(step.call_count,3);self.assertEqual(len(metrics),3)
            torch.testing.assert_close(old,original)
            self.assertGreater(float(model.scores[1].detach()),.2)
            weights.append(model.scores.detach())
        torch.testing.assert_close(weights[0],weights[1])

    def test_temperature_logprobs_match_sampling_distribution(self):
        logits = torch.tensor([[[1., 2., -1.], [2., -.5, 1.], [0., 0., 0.]]])
        model = lambda **kwargs: SimpleNamespace(logits=logits)
        tokens = torch.tensor([[0, 2, 1]])
        for temperature in (.7, 1., 1.3):
            actual = train.token_logprobs(model, tokens, torch.ones_like(tokens), 2, temperature)
            expected = (logits[:, :-1]/temperature).log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
            torch.testing.assert_close(actual, expected)

    def test_physics_starts_before_preparation_and_truncation_is_not_simulated(self):
        events = []
        inputs = SimpleNamespace(input_ids=torch.tensor([[1, 2], [1, 2]]),
                                 attention_mask=torch.ones(2, 2, dtype=torch.long))
        class Inputs(dict):
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
        model = SimpleNamespace(generate=lambda **kw: torch.tensor([[1, 2, 5, 3, 3], [1, 2, 6, 7, 8]]))
        tokenizer = SimpleNamespace(eos_token_id=3, pad_token_id=3,
                                    batch_decode=lambda *a, **k: ['finish()', 'unfinished'])
        class Pending:
            def result(self):
                self_check.assertEqual(events, ['submitted', 'prepared'])
                events.append('collected')
                return [dict(reward=1., success=True, valid=True)]
        self_check = self
        def submit(layouts, texts):
            self.assertEqual(layouts, ['a']); self.assertEqual(texts, ['finish()'])
            events.append('submitted')
            return Pending()
        with patch.object(train, 'task', SimpleNamespace(submit_batch=submit)), \
             patch.object(train, 'encode', return_value=Inputs()), patch('torch.cuda.synchronize'):
            *_, results = train.rollout(model, tokenizer, ['a', 'b'], 3, True,
                                        prepare=lambda *a: events.append('prepared'), temperature=1.2)
        self.assertEqual(events, ['submitted', 'prepared', 'collected'])
        self.assertTrue(results[0]['success'])
        self.assertEqual(results[1]['reward'], 0.)
        self.assertFalse(results[1]['physics_started'])

    def test_fresh_varied_splits_and_hidden_seed(self):
        with patch.object(task_assembly, 'DATASET_PROFILE', 'varied_v2'):
            splits = task_assembly.datasets()
        self.assertEqual([len(splits[k]) for k in ('train', 'validation', 'test')], [32, 8, 24])
        seeds = [x.seed for values in splits.values() for x in values]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(set(seeds).isdisjoint(range(41000, 44000)))
        self.assertEqual(len({task_assembly.prompt(x) for values in splits.values() for x in values}), 1)

    def test_batch_length_mismatch_fails_before_execution(self):
        with self.assertRaises(ValueError):
            task_assembly.execute_batch([task_assembly.Layout(51001)], [])


if __name__ == '__main__':
    unittest.main()
