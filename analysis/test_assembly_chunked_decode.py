"""Cache, padding, forced-echo and mode-restoration checks, not task outcomes."""
import unittest
try:
    import torch
except ImportError:
    torch=None
from analysis.assembly_chunked_decode import generate

class Grammar:
    tokens=[100,1,101,2,1,1,103,2,99];digits=[1,2]
    def allowed(self,position,row):
        groups={1:(1,3),3:(1,3),5:(5,7),7:(5,7)}
        if position in groups:return [d for d in self.digits if d not in {row[p] for p in groups[position] if p<position}]
        if position==4:return [row[1]]
        return [self.tokens[position]]
    def validate(self,row):
        assert len(row)==len(self.tokens)
        for p,t in enumerate(row):assert t in self.allowed(p,row)

@unittest.skipIf(torch is None,'Torch is checked on the GPU host')
class ChunkedTests(unittest.TestCase):
    def model(self,cache=True):
        from types import SimpleNamespace
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=[]
            def forward(self,input_ids,attention_mask,position_ids,past_key_values,use_cache,logits_to_keep):
                consumed=0 if past_key_values is None else past_key_values
                assert attention_mask.shape[1]==consumed+input_ids.shape[1]
                expected=attention_mask.long().cumsum(-1)-1;expected.masked_fill_(attention_mask==0,1)
                assert torch.equal(position_ids,expected[:,-input_ids.shape[1]:])
                assert use_cache and logits_to_keep==1
                self.calls.append(input_ids.clone())
                logits=torch.zeros((len(input_ids),1,120));logits[:,:,2]=2.
                return SimpleNamespace(logits=logits,past_key_values=attention_mask.shape[1] if cache else None)
        return Model()
    def inputs(self):return torch.tensor([[0,7,8],[6,7,8]]),torch.tensor([[0,1,1],[1,1,1]])
    def test_forced_stretches_echoes_padding_and_cache(self):
        model=self.model();ids,mask=self.inputs();out,info=generate(model,ids,mask,Grammar(),trace=True)
        self.assertEqual(out[:,3:].tolist(),[[100,2,101,1,2,2,103,1,99]]*2)
        self.assertEqual(info['forward_calls'],2);self.assertEqual(len(model.calls),2)
        self.assertEqual(model.calls[1].tolist(),[[2,101,1,2]]*2)
        self.assertEqual(len(info['choice_logprobs'][0]),2);self.assertTrue(model.training)
    def test_sampled_programs_are_legal_and_seeded(self):
        ids,mask=self.inputs();model=self.model().eval()
        a,_=generate(model,ids,mask,Grammar(),sample=True,generator=torch.Generator().manual_seed(9))
        b,_=generate(model,ids,mask,Grammar(),sample=True,generator=torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(a,b));self.assertFalse(model.training)
    def test_failure_restores_model_mode(self):
        model=self.model(cache=False);ids,mask=self.inputs()
        with self.assertRaises(ValueError):generate(model,ids,mask,Grammar())
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):generate(model,ids,mask,Grammar(),temperature=0.)

if __name__=='__main__':unittest.main()
