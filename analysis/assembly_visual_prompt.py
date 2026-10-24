"""Frozen, auditable initial-RGB conditioning for an optional preference phase."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path

PROFILE = 'initial_rgb_member_estimates_v1'
SOURCES = ('assembly_initial_observation.py', 'assembly_visual_prompt.py', 'assembly_visual_stage.py',
           'assembly_visual_observation_audit.py')


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n')


def enabled(metadata):
    profile = metadata.get('prompt_profile', 'constant')
    if profile not in ('constant', PROFILE): raise ValueError('Unknown assembly prompt profile')
    return profile == PROFILE


def verify_sources(root, metadata):
    if not enabled(metadata): return
    expected = metadata['visual_source_sha256']
    if set(expected) != set(SOURCES): raise ValueError('Incomplete visual prompt source receipt')
    for name, digest in expected.items():
        if sha(Path(__file__).with_name(name)) != digest or sha(Path(root)/'preference_source'/name) != digest:
            raise ValueError('Use the frozen visual prompt implementation: '+name)
    if 'observations_manifest_sha256' in metadata and sha(Path(root)/'observations_manifest.json') != metadata['observations_manifest_sha256']:
        raise ValueError('Development observation manifest changed')


def layout_seed(key):
    if not key.startswith('assembly-full-v1-'): raise ValueError('Unknown layout identity')
    return int(key.removeprefix('assembly-full-v1-'))


def load_observation(root, metadata, key):
    from analysis.assembly_initial_observation import describe
    if key not in {k for values in metadata['splits'].values() for k in values}:
        raise ValueError('Observation outside declared splits')
    root = Path(root); directory = root/'observations'/key
    receipt = read(directory/'receipt.json')
    required = {'camera0.png', 'camera2.png', 'camera3.png', 'sensor_inputs.json',
                'observation.json', 'prompt_fragment.txt'}
    if (set(receipt['files_sha256']) != required or receipt['seed'] != layout_seed(key) or
            receipt['observation_time_s'] != 0. or
            receipt['source_sha256'] != metadata['visual_source_sha256']['assembly_initial_observation.py'] or
            receipt['physical_provenance'] != read(root/'assembly_provenance.json')):
        raise ValueError('Observation source, scene or sensor receipt mismatch')
    for name, digest in receipt['files_sha256'].items():
        if sha(directory/name) != digest: raise ValueError('Observation payload changed: '+name)
    observation = read(directory/'observation.json')
    if observation['time_s'] != 0.: raise ValueError('Not an initial observation')
    if (directory/'prompt_fragment.txt').read_text() != describe(observation)+'\n':
        raise ValueError('Prompt fragment is not the declared visual estimate')
    manifest_name = ('test_observations_manifest.json' if key in metadata['splits']['test']
                     else 'observations_manifest.json')
    manifest = root/manifest_name
    if manifest.exists() and read(manifest)['receipts_sha256'].get(key) != sha(directory/'receipt.json'):
        raise ValueError('Observation receipt differs from frozen manifest')
    return observation


def prepare(root, metadata, split):
    """Capture train/validation before training; test only after frozen selection."""
    from analysis.assembly_initial_observation import capture
    verify_sources(root, metadata)
    if not enabled(metadata): return
    root = Path(root)
    if split not in ('development', 'test'): raise ValueError('Unknown observation collection split')
    if split == 'test':
        if not (root/'selection.json').exists():
            raise ValueError('Held-out observations require a frozen selection')
        selection = read(root/'selection.json')
        if selection['validation_success_rate'] != 1. or selection['best_step'] <= 0:
            raise ValueError('Held-out observations require a learned 8/8 validation selection')
        keys = metadata['splits']['test']; name = 'test_observations_manifest.json'
    else:
        if (root/'training.jsonl').exists(): raise ValueError('Development observations must precede optimization')
        keys = metadata['splits']['train']+metadata['splits']['validation']; name = 'observations_manifest.json'
    manifest_path = root/name
    frozen = manifest_path.exists()
    receipts = {}
    for key in keys:
        destination = root/'observations'/key
        if not destination.exists():
            if frozen: raise ValueError('Frozen observation payload is missing')
            capture(layout_seed(key), destination, read(root/'assembly_provenance.json'))
        load_observation(root, metadata, key)
        receipts[key] = sha(destination/'receipt.json')
    manifest = dict(format=PROFILE, split=split, receipts_sha256=receipts,
                    scope='Reset-time images and public sensors only; no outcome or hidden object state in policy input.')
    if frozen:
        if read(manifest_path) != manifest: raise ValueError('Frozen observation manifest changed')
    else: dump(manifest_path, manifest)


def expected_prompt(root, metadata, layout, base=None):
    from experiments.cap_rl import task_assembly as task
    from analysis.assembly_initial_observation import describe
    base = task.prompt if base is None else base
    original = base(layout)
    if not enabled(metadata): return original
    return original+'\n\n'+describe(load_observation(root, metadata, layout.key))


@contextmanager
def installed(root, metadata):
    from experiments.cap_rl import task_assembly as task
    verify_sources(root, metadata)
    original = task.prompt
    if enabled(metadata):
        task.prompt = lambda layout: expected_prompt(root, metadata, layout, base=original)
    try: yield
    finally: task.prompt = original


def augment_preferences(data, root, metadata):
    """Attach scene-matched sensor inputs to the policy's historical own samples."""
    from experiments.cap_rl.task_assembly import Layout
    result = copy.deepcopy(data)
    result['prompt_profile'] = PROFILE
    result['raw_preference_data_sha256'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    result['scope'] = ('Own historical programs re-prompted with initial RGB estimates of the same scene. '
                       'Original physical outcomes certify code/layout behavior, not sampling under the new prompt.')
    result['observations_manifest_sha256'] = sha(Path(root)/'observations_manifest.json')
    for pair in result['pairs']:
        layout = Layout(layout_seed(pair['layout_id']))
        pair['source_prompt_sha256'] = hashlib.sha256(pair['prompt'].encode()).hexdigest()
        pair['prompt'] = expected_prompt(root, metadata, layout)
        pair['observation_receipt_sha256'] = sha(Path(root)/'observations'/layout.key/'receipt.json')
    return result


def audit_observations(root, metadata, include_test=False):
    """Recompute visual estimates from saved images, never from object poses."""
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    from analysis.assembly_initial_observation import extract
    verify_sources(root, metadata); root = Path(root)
    if sha(root/'observations_manifest.json') != metadata['observations_manifest_sha256']:
        raise ValueError('Development observation manifest changed')
    cv2.setNumThreads(1)
    keys = metadata['splits']['train']+metadata['splits']['validation']
    if include_test: keys += metadata['splits']['test']
    for key in keys:
        expected = load_observation(root, metadata, key); directory = root/'observations'/key
        data = read(directory/'sensor_inputs.json'); robot = data['robot']
        for name in ('tcp','rotation','joints','joint_velocity','gripper_command'):
            robot[name] = np.array(robot[name])
        frames = [None]*4
        for i in (0,2,3): frames[i] = imageio.imread(directory/f'camera{i}.png')
        if extract(frames, data['cameras'], robot) != expected:
            raise ValueError('Saved images do not reproduce the policy observation: '+key)
    return dict(passes=True, observations_recomputed=len(keys),
                scope='Exact visual-estimate replay from saved sensor bytes; no task execution or reward.')
