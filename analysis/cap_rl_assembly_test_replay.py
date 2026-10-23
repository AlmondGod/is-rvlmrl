"""Independently replay every distinct before/after test trajectory, bypassing cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    report=json.loads((Path(args.run)/'test_comparison.json').read_text())
    rows=report['before']+report['after']
    unique={r['evidence_id']:r['evidence'] for r in rows if r.get('evidence_id')}
    def replay(pair):
        key,source=pair;out=root/key
        with (root/(key+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-u','-m','experiments.cap_rl.audit_assembly',
                '--evidence',source,'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((out/'audit.json').read_text());assert result['passes']
        print(json.dumps(dict(evidence_id=key,passes=True)),flush=True)
        return dict(evidence_id=key,**result)
    with ThreadPoolExecutor(args.workers) as pool:audits=list(pool.map(replay,unique.items()))
    result=dict(passes=True,scope='Fresh physical reruns of every distinct held-out before/after trajectory, with original result fields and all recorded states compared; no cache reuse.',
        test_records=len(rows),physical_records=sum(bool(r.get('evidence_id')) for r in rows),unique_trajectories_replayed=len(audits),
        absolute_tolerance=1e-7,test_comparison_sha256=hashlib.sha256((Path(args.run)/'test_comparison.json').read_bytes()).hexdigest(),audits=audits)
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
