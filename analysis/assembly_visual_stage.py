"""Apply frozen visual prompts to the existing final-evaluation entry points."""
import argparse
from pathlib import Path
import runpy
import sys

from analysis.assembly_visual_prompt import enabled, installed, prepare, read, verify_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--stage', choices=('primary_test','raw_baseline','unconstrained_transfer','generation_audit'), required=True)
    a = p.parse_args(); root = Path(a.run); metadata = read(root/'metadata.json')
    if not enabled(metadata): raise ValueError('Visual stage requires the declared visual prompt profile')
    verify_sources(root, metadata)
    prepare(root, metadata, 'test')
    stages = {
        'primary_test': ('analysis.cap_rl_assembly_grammar', ['test','--out',str(root)]),
        'raw_baseline': ('analysis.cap_rl_assembly_raw_baseline', ['--run',str(root)]),
        'unconstrained_transfer': ('analysis.cap_rl_assembly_unconstrained', ['--run',str(root),'--split','test']),
        'generation_audit': ('analysis.cap_rl_assembly_generation_audit', ['--run',str(root),'--out',str(root/'generation_audit.json')]),
    }
    module, arguments = stages[a.stage]
    sys.argv = [module, *arguments]
    with installed(root, metadata): runpy.run_module(module, run_name='__main__')


if __name__ == '__main__': main()
