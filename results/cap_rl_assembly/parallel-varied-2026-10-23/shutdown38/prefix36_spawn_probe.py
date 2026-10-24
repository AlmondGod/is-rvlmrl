import json
from pathlib import Path
from analysis.assembly_prefix_rewards import GroupedPrefixQueue,install
from experiments.cap_rl import task_assembly as task

def main():
    task.configure('contact_and_nails','southwest_approach','vectorized_fused')
    install(task);task.EXPECTED_PROVENANCE=task.provenance()
    queue=GroupedPrefixQueue(task,workers=1)
    try:
        rows=queue.submit([task.Layout(51001),task.Layout(51002)],['invalid()','import os']).result()
        assert len(rows)==2 and all(not r['physics_started'] and r['reward']==0 for r in rows)
        print(json.dumps(dict(passes=True,scope='Actual spawned worker initialization, frozen provenance and parser-only rewards; no physics or model.',memory_budget=queue.memory_budget,rows=rows)))
    finally:queue.close()
if __name__=='__main__':main()
