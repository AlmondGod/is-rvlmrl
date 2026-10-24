import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict
from safetensors.torch import load_file
from analysis.assembly_training_state import save, restore_optimizer_rng


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(3,2)
    def forward(self,x):
        return self.linear(x)


class TrainingStateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.model=get_peft_model(Tiny(),LoraConfig(r=2,lora_alpha=2,target_modules=['linear']))
        self.optimizer=torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=.01)
        self.patch=patch('torch.cuda.is_available',return_value=False);self.patch.start();self.addCleanup(self.patch.stop)
        self.count=patch('torch.cuda.device_count',return_value=0);self.count.start();self.addCleanup(self.count.stop)
    def update(self,optimizer):
        optimizer.zero_grad();self.model(torch.ones(2,3)).square().mean().backward();optimizer.step()
    def test_restored_adamw_and_rng_reproduce_next_cpu_update(self):
        self.update(self.optimizer)
        path=save(self.model,self.optimizer,self.root,dict(step=1))
        random_values=(random.random(),np.random.rand(),torch.rand(3))
        self.update(self.optimizer)
        expected={k:v.detach().clone() for k,v in get_peft_model_state_dict(self.model).items()}
        set_peft_model_state_dict(self.model,load_file(str(path/'adapter_model.safetensors')))
        optimizer=torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=.2)
        self.assertEqual(restore_optimizer_rng(self.model,optimizer,path),1)
        self.assertEqual(random.random(),random_values[0]);self.assertEqual(np.random.rand(),random_values[1])
        self.assertTrue(torch.equal(torch.rand(3),random_values[2]))
        self.update(optimizer)
        self.assertTrue(all(torch.equal(v,get_peft_model_state_dict(self.model)[k]) for k,v in expected.items()))
    def test_changed_payload_or_wrong_loaded_adapter_rejected(self):
        self.update(self.optimizer);path=save(self.model,self.optimizer,self.root,dict(step=1))
        self.update(self.optimizer)
        with self.assertRaisesRegex(ValueError,'Load the checkpoint adapter'):
            restore_optimizer_rng(self.model,self.optimizer,path)
        with (path/'training_state.pt').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'payload changed'):
            restore_optimizer_rng(self.model,self.optimizer,path)
    def test_never_overwrite_a_published_checkpoint(self):
        save(self.model,self.optimizer,self.root,dict(step=1))
        with self.assertRaises(FileExistsError):save(self.model,self.optimizer,self.root,dict(step=1))
        self.assertFalse((self.root/'checkpoints/.step-0001.pending').exists())


if __name__=='__main__':unittest.main()
