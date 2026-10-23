"""Finish validation selection at its proven upper bound, before any test use."""
import datetime,hashlib,json,os,pathlib,signal,time
root=pathlib.Path('/opt/cap-assembly/assembly04');pid,group,target=16142,16140,6
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
metadata=read(root/'metadata.json')
assert metadata['steps']==16 and metadata['optimization_reward']=='strict_success'
assert not (root/'test_comparison.json').exists() and not (root/'selection.json').exists()
assert not (root/f'validation_{target:04d}.json').exists()
observed=[(s,read(root/f'validation_{s:04d}.json')['metrics']) for s in (2,4)]
assert all((v['success_rate'],v['mean_reward'])==(1.,1.) for _,v in observed)
policy=dict(format='cap_rl_validation_ceiling_policy_v1',created_utc=now(),target_update=target,
 upper_bound=[1.,1.],observed_maximum_steps=[s for s,_ in observed],
 reason='Strict binary reward bounds both selection coordinates by one. Earliest-best tie handling means no future checkpoint can replace the existing first maximum. Finish the current update and scheduled validation, then test the already fixed selected model.',
 planned_updates=16,uses_test_outcomes=False)
policy_path=root/'validation_ceiling_policy.json';assert not policy_path.exists()
policy_path.write_text(json.dumps(policy,indent=2)+'\n')
print(json.dumps(policy),flush=True)
while True:
 command=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
 assert b'/opt/cap-assembly/train_assembly_strict.py' in command and str(root).encode() in command
 assert os.getpgid(pid)==group
 try:read(root/f'validation_{target:04d}.json')
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(.2);continue
 os.killpg(group,signal.SIGSTOP)
 try:
  metrics=[json.loads(s) for s in (root/'training.jsonl').read_text().splitlines()]
  assert [r['step'] for r in metrics]==list(range(1,target+1)), 'Missed exact update boundary'
  rows=[json.loads(s) for s in (root/'rollouts.jsonl').read_text().splitlines()]
  assert len(rows)==target*metadata['group_size']*metadata['prompts_per_step']
  cert=read(root/f'checkpoints/step-{target:04d}/checkpoint.json')
  for name,digest in cert['files_sha256'].items():assert sha(root/f'checkpoints/step-{target:04d}'/name)==digest
  validation=[(0,read(root/'validation_baseline.json'))]+[(s,read(root/f'validation_{s:04d}.json')) for s in range(2,target+1,2)]
  best=(-1.,-1.);best_step=None
  for step,report in validation:
   values=report['metrics'];score=(values['success_rate'],values['mean_reward'])
   assert values['n']==4 and len(report['rows'])==4
   assert values['success_rate']==sum(r['success'] for r in report['rows'])/4
   assert values['mean_reward']==sum(r['reward'] for r in report['rows'])/4
   if score>best:best,best_step=score,step
  assert best==(1.,1.) and best_step==2
  assert sha(root/'best_adapter/adapter_model.safetensors')==sha(root/'checkpoints/step-0002/adapter_model.safetensors')
  assert not (root/'test_comparison.json').exists() and not (root/'selection.json').exists()
  elapsed=float(pathlib.Path('/proc/uptime').read_text().split()[0])-int(pathlib.Path(f'/proc/{pid}/stat').read_text().split()[21])/os.sysconf('SC_CLK_TCK')
  names=['metadata.json','training.jsonl','rollouts.jsonl','validation_ceiling_policy.json','validation_baseline.json']+[f'validation_{s:04d}.json' for s in range(2,target+1,2)]
  receipt=dict(format='cap_rl_validation_ceiling_stop_v1',utc=now(),upper_bound=[1.,1.],
   planned_updates=metadata['steps'],completed_updates=target,selected_step=best_step,test_evaluated=False,
   trainer_elapsed_seconds=elapsed,files_sha256={name:sha(root/name) for name in names},
   reason=policy['reason'],source_sha256=sha(pathlib.Path(__file__)))
  (root/'validation_ceiling_stop.json').write_text(json.dumps(receipt,indent=2)+'\n')
  selection=dict(best_step=best_step,validation_success_rate=best[0],validation_mean_reward=best[1],
   selection_split='validation',training_seconds=metrics[-1]['elapsed_seconds'],
   training_seconds_scope='Training loop through last completed optimizer update, including prior periodic validation and CPU physics waits; excludes setup, baseline validation, terminal validation, final test and audits. End-to-end allocation is reported separately.',
   stopping_rule='validation_ceiling_stop.json')
  temporary=root/'selection.json.partial';temporary.write_text(json.dumps(selection,indent=2)+'\n');temporary.rename(root/'selection.json')
  os.killpg(group,signal.SIGTERM);os.killpg(group,signal.SIGCONT)
  print(json.dumps(receipt),flush=True)
 except BaseException:
  os.killpg(group,signal.SIGCONT);raise
 break
