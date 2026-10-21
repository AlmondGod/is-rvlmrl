"""Prepare a portable cloud bundle without running or compiling physics."""
import hashlib
import json
from pathlib import Path
import tarfile
from .design import ROOT


def main():
    files=[p for p in (ROOT/'experiments/box_assembly').rglob('*') if p.is_file() and p.suffix in ('.py','.json','.md','.sh','.csv')]
    files+=[ROOT/'experiments/astra_camera/requirements.txt',ROOT/'third_party/wuji-description/LICENSE']
    files+=[p for p in (ROOT/'third_party/wuji-description/hand2/hand2_2/body').rglob('*') if p.is_file()]
    out=ROOT/'outputs/box_assembly';out.mkdir(parents=True,exist_ok=True)
    manifest=out/'bundle_manifest.json'
    manifest.write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    bundle=out/'cloud_bundle.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        for p in files+[manifest]:archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    print(json.dumps({'bundle':str(bundle),'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
