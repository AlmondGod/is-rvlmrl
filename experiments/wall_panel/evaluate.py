"""Freeze before reserved seeds; retain failures; never tune or retry this set."""
import argparse,json,hashlib,shutil,subprocess,sys
from pathlib import Path

def sources():
 return [p for folder in ['experiments/wall_panel','experiments/frame_assembly_v2','experiments/arm_corner'] for p in Path(folder).glob('*') if p.suffix in ('.py','.json','.txt')]
def digest():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);frozen=digest();(out/'freeze.json').write_text(json.dumps({'sources':frozen,'protocol':json.loads(Path('experiments/wall_panel/protocol.json').read_text())},indent=2))
 for p in sources():dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 results=[]
 for name,seed,dt in [('half_step',0,.0005),('reserved_9400',9400,.001),('reserved_9401',9401,.001),('reserved_9402',9402,.001)]:
  if digest()!=frozen:raise RuntimeError('Source changed after freeze; refusing remaining reserved evaluation')
  with (out/(name+'.log')).open('w') as log:rc=subprocess.call([sys.executable,'-m','experiments.wall_panel.run','--out',str(out/name),'--seed',str(seed),'--dt',str(dt)],stdout=log,stderr=subprocess.STDOUT)
  p=out/name/'result.json';results.append({'name':name,'returncode':rc,'result':json.loads(p.read_text()) if p.exists() else None});(out/'summary.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
