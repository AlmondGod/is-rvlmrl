"""Package whitelisted code/assets for upload; never runs the simulator."""
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    root=Path(__file__).resolve().parents[2]
    files=list((root/'experiments/astra_camera').glob('*.py'))
    files += [root/'experiments/astra_camera'/n for n in ['requirements.txt','cloud_run.sh','README.md']]
    files += [root/p for p in ['experiments/frame_assembly/environment.py','experiments/frame_assembly/rail_rl.py',
        'experiments/astra_closed_loop/harness.py','outputs/frame_assembly/rl/base_params.json',
        'outputs/frame_assembly/astra/grasp_targets.json']]
    files += [p for p in (root/'third_party/wuji-description/hand2/hand2_2/body').rglob('*') if p.is_file()]
    for name in ['controller.py','perception.py']:
        if not (root/'experiments/astra_camera'/name).is_file():raise FileNotFoundError(name)
    out=root/'outputs/frame_assembly/astra_camera';out.mkdir(parents=True,exist_ok=True)
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path=out/'bundle_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    target=out/'cloud_bundle.tar.gz'
    with tarfile.open(target,'w:gz') as bundle:
        for p in files+[manifest_path]:bundle.add(p,arcname=str(p.relative_to(root)),recursive=False)
    print(json.dumps({'bundle':str(target),'bytes':target.stat().st_size,'files':len(files),
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'simulations_run':0}))


if __name__=='__main__':main()
