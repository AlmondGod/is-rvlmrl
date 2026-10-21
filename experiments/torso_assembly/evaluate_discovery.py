"""Evaluator-only batch for RGB discovery. Never exposes seeds/GT to observer."""
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco
from .environment import Env
from .discovery import discover
from experiments.arm_corner.run import clean

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);rows=[]
 for seed in args.seeds:
  e=Env(layout_seed=seed);renderer=mujoco.Renderer(e.m,960,1280);frames,cameras=e.images(renderer);start=time.monotonic();est,diag=discover(frames,cameras,e.robot());elapsed=time.monotonic()-start
  errors=None if est is None else [float(np.linalg.norm(est[i]['center']-e.d.xpos[e.woods[i]])*1000) for i in range(4)]
  row=dict(seed_evaluator_only=seed,accepted=est is not None,passed=errors is not None and max(errors)<=6,center_errors_mm=errors,bootstrap_wall_s=elapsed,diagnostics=clean(diag));rows.append(row);renderer.close();print(json.dumps(row),flush=True)
  (out/'results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
