"""Create a content-addressed, self-contained runtime bundle and verify it.

Copies source and robot assets, never datasets, credentials, or past episode files.
Execute from bundle cwd with an absolute Python interpreter and -B.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

PACKAGES = ("assembly_benchmark", "torso_assembly", "frame_assembly_v2",
            "wall_panel", "arm_corner", "astra_camera", "box_assembly")
ASSETS = ("third_party/mujoco_menagerie/universal_robots_ur10e",
          "third_party/3-PoinTr/simulation_assets/ufactory_xarm7")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def content_manifest(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and
            p.name not in ("manifest.json", "freeze.json")}

def verify(root):
    root = Path(root)
    frozen = json.loads((root / "manifest.json").read_text())
    actual = content_manifest(root)
    changed = [name for name in sorted(frozen.keys() | actual.keys())
               if frozen.get(name) != actual.get(name)]
    if changed:
        raise RuntimeError("Frozen bundle changed: " + repr(changed[:20]))
    recorded = json.loads((root / "freeze.json").read_text())
    if digest(root / "manifest.json") != recorded["manifest_sha256"]:
        raise RuntimeError("Manifest hash mismatch")
    versions_path = root / "requirements-frozen.json"
    if versions_path.exists():
        for name, expected in json.loads(versions_path.read_text()).items():
            try:
                actual_version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual_version = None
            if actual_version != expected:
                raise RuntimeError("Runtime version differs: " + name)
    if recorded.get("python", platform.python_version()) != platform.python_version():
        raise RuntimeError("Python version differs from freeze.")
    return recorded

def create(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in PACKAGES:
        shutil.copytree(Path("experiments") / name, root / "experiments" / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for asset in ASSETS:
        shutil.copytree(asset, root / asset, ignore=shutil.ignore_patterns("__pycache__"))
    versions = {}
    for name in ("mujoco", "numpy", "scipy", "opencv-python", "imageio", "imageio-ffmpeg"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    (root / "requirements-frozen.json").write_text(json.dumps(versions, indent=2))
    (root / "manifest.json").write_text(json.dumps(content_manifest(root), indent=2))
    protocol = json.loads((root / "experiments/assembly_benchmark/protocol.json").read_text())
    metadata = {"manifest_sha256": digest(root / "manifest.json"),
                "python": platform.python_version(), "platform": platform.platform(),
                "source_files_and_assets": len(content_manifest(root)),
                "physical_validation": False, "no_runpod": protocol["no_runpod"]}
    (root / "freeze.json").write_text(json.dumps(metadata, indent=2))
    # Read-only files deter accidental edits; hashes are the actual integrity gate.
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return verify(root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle) if args.verify else create(args.bundle), indent=2))
