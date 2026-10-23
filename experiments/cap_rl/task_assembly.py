"""Offline full-program CaP-RL adapter for the three-arm physical benchmark."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from .assembly import parse_program, provenance, run

SYSTEM = 'Write only the requested Python API program. Do not explain it.'
EXPECTED_PROVENANCE = None


@dataclass(frozen=True)
class Layout:
    seed: int

    @property
    def key(self):
        return f'assembly-full-v1-{self.seed}'


def datasets():
    # Fresh layouts, distinct from historical development and model pilots.
    # The policy does not receive the seed. The same program must work through
    # supplied RGB feedback skills under lane, XY and yaw randomization.
    return {'train': [Layout(i) for i in range(41001, 41009)],
            'validation': [Layout(i) for i in range(42001, 42005)],
            'test': [Layout(i) for i in range(43001, 43009)]}


def prompt(layout):
    return '''Write a complete program for three robot arms to assemble a wooden frame with plywood and flip it.
The four wooden members and plywood start loose. Two arms have grippers and the third has a nail tool. Each API call runs a supplied camera-guided feedback skill to completion. Grasping uses physical contacts; connections are finite-strength simulated nails. Stop automatically on any failed skill. Six seconds of physical settling occur before each call.
Member identities: 0 long left-side member -> front; 1 short left-side member -> left; 2 long right-side member -> rear; 3 short right-side member -> right.
API: only these calls, with literal integer indices 0..3 where specified:
position_member(i): pick, lift, carry and align member i; keeps holding it.
release_member(i): release the positioned member and withdraw; one release per member.
park_grippers(): after all four members are released, park both arms clear of the nail tool.
nail_corner(i): visit corner i once (0 southwest, 1 southeast, 2 northeast, 3 northwest); requires all four members released and grippers parked.
feed_panel(): after four corner visits, feed the plywood sheet into the workspace.
position_panel(): after feeding, grasp the sheet with both grippers, lift it off its supports, retract shelves, align and lower; keeps holding it.
release_panel(): release the positioned sheet and withdraw.
nail_panel(i): visit the sheet-to-member i connection once; requires sheet released.
flip_structure(): after all eight nail visits, grasp, lift clear, turn 180 degrees and hold inverted.
finish(): declare completion; must be the last call.
stop(): end an unsuccessful attempt.
All four members must form a released frame before corner nailing. The sheet must be placed and released before sheet nailing. Success requires all eight connections intact, correct frame/sheet geometry, release and stability before lifting, and the entire structure retained during a five-second inverted hold. Triggering nails alone does not ensure success.
Output only API calls, at most 32. No loops, variables, imports, functions, calculations, prints, or explanations.'''


def execute(layout, text):
    try:
        calls = parse_program(text)
    except (SyntaxError, ValueError) as exc:
        return dict(reward=0., success=False, valid=False, error=str(exc), physics_started=False)
    # Exact-program memoization only. No predicted rewards, prefix surrogate,
    # reset-state injection, or transfer between different simulator versions.
    manifest = provenance()
    if EXPECTED_PROVENANCE is not None and manifest != EXPECTED_PROVENANCE:
        raise RuntimeError('Assembly dependencies changed during this training/evaluation run')
    identity = dict(seed=layout.seed, calls=calls, provenance=manifest)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(os.environ.get('CAP_RL_ASSEMBLY_CACHE', 'artifacts/assembly_rollouts')).resolve()
    out = root / key
    certificate = out / 'certificate.json'
    reused = certificate.exists()
    if reused:
        cert = json.loads(certificate.read_text())
        for name, sha in cert['files'].items():
            if hashlib.sha256((out / name).read_bytes()).hexdigest() != sha:
                raise RuntimeError('Cached assembly evidence changed: ' + name)
        if cert['identity_sha256'] != key:
            raise RuntimeError('Cached assembly identity mismatch')
        result = json.loads((out / 'result.json').read_text())
    else:
        # Existing incomplete runs are errors, never silently retried/relabelled.
        result = run(layout.seed, text, out)
        cert = dict(identity_sha256=key, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in out.iterdir() if p.is_file()})
        certificate.write_text(json.dumps(cert, indent=2))
    summary = {k: v for k, v in result.items() if k not in ('history', 'final_state', 'last_flip')}
    summary.update(evidence=str(out), evidence_id=key, cache_hit=reused,
                   result_sha256=cert['files']['result.json'])
    return summary
