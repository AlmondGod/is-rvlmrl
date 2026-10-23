import hashlib,json,os,pathlib
from experiments.cap_rl import task_assembly as task
if __name__=='__main__':
    root=pathlib.Path('/opt/cap-assembly')
    out=root/'assembly03-training-prefetch.json'
    assert not out.exists()
    os.environ['CAP_RL_ASSEMBLY_CACHE']=str(root/'assembly03-evidence')
    task.configure('contact_and_nails','southwest_approach')
    task.EXPECTED_PROVENANCE=task.provenance()
    task.PHYSICS_WORKERS=4
    source=root/'assembly02/rollouts.jsonl'
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    original=next(r for r in rows if r['step']==1 and r.get('seed')==41001 and r.get('fasteners',{}).get('intact')==7)
    text=original['completion'];seeds=[41002,41006,41005,41007,41003]
    result=dict(scope='Speculative training-layout execution of an already independently sampled program. No targets or demonstrations supplied to the learner; cache reuse requires an exact independently sampled AST and matching layout/provenance.',
        source_rollout_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_step=original['step'],source_evidence_id=original['evidence_id'],
        seeds=seeds,program=text,provenance=task.EXPECTED_PROVENANCE,
        results=task.execute_batch([task.Layout(s) for s in seeds],[text]*len(seeds)))
    out.write_text(json.dumps(result,indent=2)+'\n')
    task._POOL.shutdown()
    print('Training-layout prefetch complete',flush=True)
