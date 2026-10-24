import argparse,hashlib,json,pathlib,shutil,time
import cv2
from experiments.torso_assembly.acceleration import configure
from experiments.cap_rl import assembly
from analysis.compare_assembly_speed import compare
import native_substep18 as candidate
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--out',required=True);a=p.parse_args()
base=pathlib.Path(a.baseline);out=pathlib.Path(a.out)
if out.exists() or out.with_suffix('.sources').exists():raise RuntimeError('Preserve previous probe')
r=json.loads((base/'result.json').read_text());cv2.setNumThreads(1);configure('vectorized_native')
receipt=candidate.install();receipt.update(baseline_result_sha256=hashlib.sha256((base/'result.json').read_bytes()).hexdigest(),scope='Experimental CPU inner-step fusion; unchanged physical task and no training-cache use.')
sources=out.with_suffix('.sources');sources.mkdir()
for name in ('native_substep18.py','native_substep18.c','run_native_substep18.py'):
 shutil.copyfile(pathlib.Path(__file__).with_name(name),sources/name)
receipt['sources']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in sources.iterdir()}
out.with_suffix('.candidate.json').write_text(json.dumps(receipt,indent=2)+'\n')
original=assembly.provenance
assembly.provenance=lambda **kwargs:dict(original(**kwargs),experimental_candidate=receipt)
result=assembly.run(r['seed'],(base/'program.py').read_text(),out,support_profile='contact_and_nails',controller_profile='southwest_approach')
report=compare(base,out);out.with_suffix('.comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if not report['exact']:raise SystemExit(1)
