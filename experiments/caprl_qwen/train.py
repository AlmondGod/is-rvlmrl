"""Qwen2.5-Coder 0.5B LoRA: expert SFT, then physics-reward policy gradient."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model
from .environment import Push,expert

def sample(model,tok,prompt,group=1,greedy=False):
    prefix=tok(prompt,return_tensors="pt").to(model.device)
    with torch.no_grad():
        seq=model.generate(**prefix,max_new_tokens=30,
           do_sample=not greedy,temperature=None if greedy else .8,
           top_p=None if greedy else .95,num_return_sequences=group,
           pad_token_id=tok.eos_token_id)
    start=prefix.input_ids.shape[1]
    return seq,start,[tok.decode(row[start:],skip_special_tokens=True) for row in seq]

def eval_model(model,tok,seeds):
    model.eval()
    env=Push()
    records=[]
    for seed in seeds:
        prompt=env.reset(seed)
        _,_,codes=sample(model,tok,prompt,greedy=True)
        reward,valid,success=env.execute(codes[0])
        records.append(dict(seed=seed,code=codes[0],reward=reward,valid=valid,success=success))
    return dict(success_rate=np.mean([r["success"] for r in records]).item(),
                valid_rate=np.mean([r["valid"] for r in records]).item(),records=records)

def token_logp(model,seq,start):
    logits=model(seq[None,:]).logits[0,start-1:-1].float()
    return torch.log_softmax(logits,dim=-1).gather(-1,seq[start:,None]).sum()

def main(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,
        target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
    holdout=list(range(100000,100000+args.eval_episodes))
    def evaluate(name):
        result=eval_model(model,tok,holdout)
        (folder/(name+".json")).write_text(json.dumps(result,indent=2))
        print(name,{k:v for k,v in result.items() if k!="records"},flush=True)
    evaluate("baseline")
    for step in range(args.sft_steps):
        prompt,code=expert(int(np.random.randint(0,10000)))
        full=tok(prompt+code+tok.eos_token,return_tensors="pt").to(model.device)
        start=tok(prompt,return_tensors="pt").input_ids.shape[1]
        labels=full.input_ids.clone();labels[:,:start]=-100
        model.train();optimizer.zero_grad(set_to_none=True)
        model(**full,labels=labels).loss.backward()
        optimizer.step()
    evaluate("sft")
    env=Push()
    for step in range(args.rl_steps):
        seed=int(np.random.randint(0,10000));prompt=env.reset(seed)
        model.eval()
        seq,start,codes=sample(model,tok,prompt,group=args.group)
        rewards=[]
        for code in codes:
            env.reset(seed)
            reward,_,_=env.execute(code)
            rewards.append(reward)
        r=torch.tensor(rewards,dtype=torch.float32)
        advantage=(r-r.mean())/(r.std(unbiased=False)+1e-4)
        model.train();optimizer.zero_grad(set_to_none=True)
        losses=[-a.to(model.device)*token_logp(model,s,start)
                for a,s in zip(advantage,seq)]
        loss=torch.stack(losses).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%10==0:
            print("rl",step+1,"mean reward",float(r.mean()),flush=True)
    evaluate("rl")
    model.save_pretrained(folder/"adapter");tok.save_pretrained(folder/"adapter")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    p.add_argument("--out",default="runs/caprl_qwen")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--sft-steps",type=int,default=80)
    p.add_argument("--rl-steps",type=int,default=100)
    p.add_argument("--eval-episodes",type=int,default=20)
    p.add_argument("--group",type=int,default=6)
    p.add_argument("--lr",type=float,default=5e-5)
    main(p.parse_args())
