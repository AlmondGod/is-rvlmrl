"""Build within-layout preferences from the policy's certified training samples.

No validation/test outcomes or externally authored programs are training data.
Pairs express strict success over failure for the SAME randomized layout. This
is an optional offline data preparation tool, not an optimizer or task result.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(run):
    run = Path(run)
    metadata = json.loads((run/'metadata.json').read_text())
    if (metadata.get('task') != 'assembly_full' or
        metadata.get('assembly_split') != 'varied_v2' or
        metadata.get('optimization_reward') != 'strict_success'):
        raise ValueError('Expected strict varied-placement assembly training')
    allowed = set(metadata['splits']['train'])
    if allowed & (set(metadata['splits']['validation']) | set(metadata['splits']['test'])):
        raise ValueError('Overlapping splits')
    source = (run/'rollouts.jsonl').read_bytes()
    provenance = json.loads((run/'assembly_provenance.json').read_text())
    rows = [json.loads(line) for line in source.splitlines() if line]
    unique = defaultdict(dict)
    certificates = {}
    for index, row in enumerate(rows):
        if row['layout_id'] not in allowed:
            raise ValueError('Non-training sample in preference inputs')
        if type(row['success']) is not bool or row['reward'] != float(row['success']):
            raise ValueError('Preference must use binary strict physical success')
        # This constrained-policy phase requires a real physical certificate for
        # every preference, including the losing sample. No synthetic negatives.
        evidence = Path(row['evidence'])
        cert = json.loads((evidence/'certificate.json').read_text())
        if evidence.name != row['evidence_id'] or cert['identity_sha256'] != row['evidence_id']:
            raise ValueError('Physical identity mismatch')
        if row['evidence_id'] not in certificates:
            for name, digest in cert['files'].items():
                if sha(evidence/name) != digest:
                    raise ValueError('Physical evidence changed: '+str(evidence/name))
            certificates[row['evidence_id']] = sha(evidence/'certificate.json')
        raw = json.loads((evidence/'result.json').read_text())
        if (raw['success'] != row['success'] or raw['passes'] != row['success'] or
            cert['files']['result.json'] != row['result_sha256'] or row.get('truncated')):
            raise ValueError('Reward/result mismatch')
        from experiments.cap_rl.assembly import parse_program
        from experiments.cap_rl.task_assembly import terminal_failure_prefix, prompt, Layout
        calls = parse_program(row['completion'])
        measured = parse_program((evidence/'program.py').read_text())
        if json.loads((evidence/'provenance.json').read_text()) != provenance:
            raise ValueError('Physical provenance differs from parent run')
        identity = dict(seed=raw['seed'], calls=measured, provenance=provenance)
        if hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest() != row['evidence_id']:
            raise ValueError('Certificate not bound to program, seed and provenance')
        if calls != measured and (row.get('cache_reuse') != 'terminal_failure_prefix' or
                                   terminal_failure_prefix(raw,calls) is None):
            raise ValueError('Unjustified failure-prefix reuse')
        if row['layout_id'] != f"assembly-full-v1-{raw['seed']}":
            raise ValueError('Physical result from another layout')
        if row['prompt'] != prompt(Layout(raw['seed'])):
            raise ValueError('Training prompt changed')
        key = json.dumps(calls)
        old = unique[row['layout_id']].get(key)
        if old and (old['success'] != row['success'] or old['prompt'] != row['prompt']):
            raise ValueError('Contradictory duplicate training outcome')
        if old is None:
            unique[row['layout_id']][key] = dict(row, source_row=index)
    pairs, layout_counts = [], {}
    for layout, programs in sorted(unique.items()):
        good = [r for r in programs.values() if r['success']]
        bad = [r for r in programs.values() if not r['success']]
        layout_counts[layout] = dict(successes=len(good), failures=len(bad), pairs=len(good)*len(bad))
        for chosen in good:
            for rejected in bad:
                if chosen['prompt'] != rejected['prompt']:
                    raise ValueError('Preference pair has different prompts')
                pairs.append(dict(layout_id=layout, prompt=chosen['prompt'],
                    chosen=chosen['completion'], rejected=rejected['completion'],
                    chosen_row=chosen['source_row'], rejected_row=rejected['source_row'],
                    chosen_evidence_id=chosen['evidence_id'], rejected_evidence_id=rejected['evidence_id']))
    return dict(format='assembly_certified_preferences_v1',
        scope='Offline preparation only. Same-layout strict success/failure pairs from self-generated training programs.',
        parent_run=str(run.resolve()), metadata_sha256=sha(run/'metadata.json'),
        rollouts_sha256=hashlib.sha256(source).hexdigest(), source_rows=len(rows),
        physical_provenance_sha256=sha(run/'assembly_provenance.json'),
        certificate_sha256=certificates, layout_counts=layout_counts, pairs=pairs)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();destination=Path(a.out)
    if destination.exists(): raise FileExistsError(destination)
    result=build(a.run);destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('source_rows','layout_counts')}),flush=True)


if __name__=='__main__':main()
