"""Compare a fresh program with the same program branching from a sibling.

All programs are supplied before simulation. This experiment never exposes
native snapshot state to a policy or treats a prefix as successful assembly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True);parser.add_argument('--branch',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--prefix-calls',type=int,default=9)
    parser.add_argument('--seconds',type=float,default=1000.)
    parser.add_argument('--backend',default='vectorized')
    args=parser.parse_args()
    from experiments.cap_rl import assembly
    from experiments.torso_assembly.acceleration import configure
    from analysis.assembly_prefix_evidence import EvidenceBoundPrefixRunner as PrefixRunner
    from analysis.compare_assembly_speed import compare
    import cv2
    cv2.setNumThreads(1);configure(args.backend)
    base,branch=[Path(p).read_text() for p in (args.base,args.branch)]
    ca,cb=map(assembly.parse_program,(base,branch))
    if (args.prefix_calls<1 or min(len(ca),len(cb))<=args.prefix_calls or
        ca[:args.prefix_calls]!=cb[:args.prefix_calls] or ca==cb):
        raise ValueError('Expected one identical prefix and distinct unexecuted suffixes')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    profiles=dict(support_profile='contact_and_nails',controller_profile='southwest_approach')
    provenance=lambda:assembly.provenance(**profiles)
    runner=PrefixRunner(assembly.run,provenance,args.prefix_calls)
    timings={}
    for name,execute,text in [('fresh',assembly.run,branch),('origin',runner,base),('reused',runner,branch)]:
        started=time.monotonic()
        execute(args.seed,text,out/name,seconds=args.seconds,**profiles)
        timings[name]=time.monotonic()-started
    receipt=json.loads((out/'reused/prefix_receipt.json').read_text())
    comparison=compare(out/'fresh',out/'reused')
    passed=bool(receipt['reused'] and comparison['exact'] and receipt.get('origin_files_sha256'))
    report=dict(passes=passed,configuration=vars(args),comparison=comparison,
                seconds=timings,reuse_receipt=receipt,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Exact branch replay experiment only, not installed in training; timing includes a separately reported cold origin episode.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
