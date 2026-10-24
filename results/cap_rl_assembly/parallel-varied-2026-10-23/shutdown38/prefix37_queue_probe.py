import json,time
from pathlib import Path
from analysis.assembly_prefix_rewards import GroupedPrefixQueue,install,verify_episode_prefix
from analysis.compare_assembly_speed import compare
from experiments.cap_rl import task_assembly as task

def main():
    task.configure('contact_and_nails','southwest_approach','vectorized_fused')
    install(task);task.EXPECTED_PROVENANCE=task.provenance()
    roots=[Path('/opt/cap-speed/fused-validation31/success'),Path('/opt/cap-speed/prefix-validation35/episodes/fresh')]
    texts=[(p/'program.py').read_text() for p in roots]
    queue=GroupedPrefixQueue(task,workers=1);started=time.monotonic()
    try:rows=queue.submit([task.Layout(51002)]*2,texts).result()
    finally:queue.close()
    comparisons=[]
    for base,row in zip(roots,rows):
        assert not row['cache_hit'],row
        comparison=compare(base,row['evidence'])
        assert comparison['exact'],comparison
        receipt=verify_episode_prefix(row['evidence'],task.provenance()['execution_prefix'])
        comparisons.append(dict(comparison=comparison,prefix_receipt=receipt))
    assert comparisons[1]['prefix_receipt']['reused'] is True
    report=dict(passes=True,rows=rows,comparisons=comparisons,wall_s=time.monotonic()-started,
        memory_admission=queue.memory_budget,scope='Actual spawned certified grouped reward queue, full physical outcome/state exactness, nine-call prefix. Baselines independently recorded; no model or optimizer.')
    Path('/opt/cap-speed/prefix-validation37/report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
