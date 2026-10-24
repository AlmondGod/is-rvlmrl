"""Bind each resident prefix to the original completed episode's evidence.

The physical snapshot implementation is unchanged. These additional receipts
make origin-file mutation/missing evidence a hard infrastructure error. They
do not replace the independent continuous-versus-restored physics checks.
"""
import gc
import json
from pathlib import Path

from analysis.assembly_prefix_runner import PrefixRunner,digest
from analysis.benchmark_assembly_prefix import SnapshotFailure


def verify_origin(entry):
    files=entry.get('origin_files_sha256')
    required={'program.py','provenance.json','result.json','states.npz','prefix_receipt.json'}
    if files is None or set(files)!=required:
        raise SnapshotFailure('Prefix origin lacks complete evidence bindings')
    root=Path(entry['origin'])
    for name,expected in files.items():
        if not (root/name).is_file() or digest(root/name)!=expected:
            raise SnapshotFailure('Prefix origin evidence changed: '+name)


class EvidenceBoundPrefixRunner(PrefixRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.sources[str(Path(__file__))]=digest(__file__)

    def __call__(self,seed,text,out,**kwargs):
        if self.entry is not None:verify_origin(self.entry)
        # Finished environments contain native/Python cycles. Reclaim them
        # before constructing the next world, retaining only the cache entry.
        gc.collect()
        result=super().__call__(seed,text,out,**kwargs)
        path=Path(out)/'prefix_receipt.json'
        if not path.exists():return result  # Parser failure, no physical prefix.
        receipt=json.loads(path.read_text())
        if receipt['reused']:
            verify_origin(self.entry)
            receipt['origin_files_sha256']=dict(self.entry['origin_files_sha256'])
        elif 'captured_step' in receipt:
            files={name:digest(Path(out)/name) for name in
                   ('program.py','provenance.json','result.json','states.npz')}
            receipt['origin_files_sha256']=files
        receipt['origin_binding_scope']='Completed origin program, provenance, result, states and capture receipt; no predicted physical reward.'
        path.write_text(json.dumps(receipt,indent=2)+'\n')
        if not receipt['reused'] and 'captured_step' in receipt:
            self.entry['origin_files_sha256']={**files,'prefix_receipt.json':digest(path)}
            verify_origin(self.entry)
        return result
