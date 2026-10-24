"""Rebuild the interrupted optimizer by replaying its exact cached first batch."""
import hashlib,json,shutil,sys
from pathlib import Path
from analysis import train_assembly_visual_grpo as online
from experiments.cap_rl import checkpoints
OLD=Path('/opt/cap-speed/varied-online13')
SOURCE=Path(__file__).resolve()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
original_optimize=online.optimize
original_save=checkpoints.save_step_adapter

def save(model,out,metric):
 original_save(model,out,metric)
 if metric['step']!=1:return
 expected=json.loads((OLD/'interruption.json').read_text())['checkpoint_sha256']
 actual=sha(Path(out)/'checkpoints/step-0001/adapter_model.safetensors')
 fields=('layout_id','prompt','completion','reward','success','evidence_id','result_sha256')
 old=[json.loads(x) for x in (OLD/'rollouts.jsonl').read_text().splitlines()]
 new=[json.loads(x) for x in (Path(out)/'rollouts.jsonl').read_text().splitlines()]
 rows_match=len(old)==len(new)==16 and all(all(a.get(k)==b.get(k) for k in fields) for a,b in zip(old,new))
 report=dict(passes=rows_match and actual==expected,expected_weights_sha256=expected,actual_weights_sha256=actual,
  first_batch_records_equal=rows_match,source_sha256=sha(SOURCE),scope='Restarted from the original warm adapter and original seed; regenerated the first programs, verified real cached rewards, and replayed four AdamW updates. Optimizer is reconstructed, not approximated from adapter-only state.')
 (Path(out)/'reconstruction_verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(stage='reconstruction',**report)),flush=True)
 if not report['passes']:raise RuntimeError('Deterministic checkpoint reconstruction failed; do not continue silently')

def optimize(metadata,out,trainer,grammar):
 archive=Path(out)/'interrupted_online_attempt';archive.mkdir()
 for name in ('metadata.json','process_start.json','training.jsonl','rollouts.jsonl','interruption.json','validation_baseline.json'):
  shutil.copyfile(OLD/name,archive/name)
 shutil.copyfile(SOURCE,Path(out)/'reconstruction_source.py')
 metadata['reconstruction']=dict(source_sha256=sha(SOURCE),interrupted_attempt_sha256={p.name:sha(p) for p in archive.iterdir()},
  checkpoint_sha256=json.loads((OLD/'interruption.json').read_text())['checkpoint_sha256'],
  recovery_method='exact_first_batch_and_AdamW_update_replay',prior_updates=4,
  resource_policy='No overlapping benchmark, prefetch, decoder model, or second learner while this run is active; eight physical workers only.')
 (Path(out)/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
 return original_optimize(metadata,out,trainer,grammar)

checkpoints.save_step_adapter=save
online.optimize=optimize
if __name__=='__main__':online.main()
