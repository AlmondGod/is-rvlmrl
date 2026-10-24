from pathlib import Path
import os,signal,time,json,hashlib
from datetime import datetime,timezone
root=Path('/opt/cap-speed');out=root/'shutdown38';out.mkdir(exist_ok=False)
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False
def children(pid):
 try:items=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
 except FileNotFoundError:return []
 return [int(p) for p in items]+[c for p in items for c in children(int(p))]
def send(pid,sig):
 try:os.kill(pid,sig);return True
 except ProcessLookupError:return False
receipt=dict(format='user_requested_shutdown_v1',started_utc=datetime.now(timezone.utc).isoformat(),reason='User reports imminent pod shutdown and requests summary/commit. No further training or validation selection.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
receipt['stopped_supervisors']={str(pid):send(pid,signal.SIGTERM) for pid in (20502,19963,18161)}
learner=[18156]+children(18156);probe=[20055]+children(20055)
receipt.update(learner_pids=learner,probe_pids=probe)
for pid in learner:send(pid,signal.SIGINT)
for pid in probe:send(pid,signal.SIGTERM)
for _ in range(15):
 if not any(alive(p) for p in learner+probe):break
 time.sleep(1)
receipt['forced_termination']=[p for p in learner+probe if alive(p)]
for pid in receipt['forced_termination']:send(pid,signal.SIGTERM)
time.sleep(2)
for pid in learner+probe:
 if alive(pid):send(pid,signal.SIGKILL)
receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(),selection_exists=(root/'varied-online25/selection.json').exists(),validation4_exists=(root/'varied-online25/validation_0004.json').exists(),test_ever_started=(root/'varied-online25/test_comparison.json').exists(),memory_events=Path('/sys/fs/cgroup/memory.events').read_text())
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
