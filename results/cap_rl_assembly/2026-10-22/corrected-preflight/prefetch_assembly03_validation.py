"""Speculative validation physics only; never supplies training rewards or targets."""
import hashlib,json,os,pathlib
from experiments.cap_rl import task_assembly as task
if __name__=='__main__':
    root=pathlib.Path('/opt/cap-assembly');out=root/'assembly03-validation-prefetch.json'
    assert not out.exists()
    os.environ['CAP_RL_ASSEMBLY_CACHE']=str(root/'assembly03-evidence')
    task.configure('contact_and_nails','southwest_approach');task.EXPECTED_PROVENANCE=task.provenance();task.PHYSICS_WORKERS=4
    source=root/'assembly02/rollouts.jsonl';rows=[json.loads(s) for s in source.read_text().splitlines()]
    original=next(r for r in rows if r['step']==1 and r.get('seed')==41001 and r.get('fasteners',{}).get('intact')==7)
    seeds=list(range(42001,42005));program=original['completion']
    result=dict(scope='Validation-only prefetch of an independently sampled program. No test layouts. No controller tuning from these outcomes. A scheduled checkpoint must independently generate this exact AST to reuse measured evidence; validation never enters optimizer batches.',
        source_rollout_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_evidence_id=original['evidence_id'],seeds=seeds,
        program=program,provenance=task.EXPECTED_PROVENANCE,
        results=task.execute_batch([task.Layout(s) for s in seeds],[program]*len(seeds)))
    out.write_text(json.dumps(result,indent=2)+'\n');task._POOL.shutdown()
    print('Validation-only prefetch complete; outcomes reserved for scheduled checkpoint evaluation.',flush=True)
