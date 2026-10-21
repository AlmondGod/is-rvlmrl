"""Reconstruct the exact historical v9 bundle without duplicating it in Git.

This checks content, not installed package versions. The runner checks both.
No downloads, model access, physics execution, or credential handling occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def restore(destination, snapshot="v9"):
    repo = Path(__file__).resolve().parents[1]
    provenance = repo / "provenance" / ("frozen_" + snapshot)
    manifest_path = provenance / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads((provenance / "freeze.json").read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(manifest_path) != metadata["manifest_sha256"]:
        raise ValueError("Historical manifest hash mismatch")
    sources = {}
    for relative, expected in manifest.items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Unsafe manifest path")
        override = provenance / "overrides" / name
        source = provenance / name if relative == "requirements-frozen.json" else (
            override if override.is_file() else repo / name)
        if not source.is_file() or digest(source) != expected:
            raise ValueError(f"Historical source changed or missing: {relative}")
        sources[relative] = source
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for relative, source in sources.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ("manifest.json", "freeze.json"):
        shutil.copyfile(provenance / name, destination / name)
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({"bundle": str(destination), "files": len(sources),
                      "manifest_sha256": metadata["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New directory; never overwritten")
    parser.add_argument("--snapshot", choices=("v8", "v9"), default="v9")
    args = parser.parse_args()
    restore(args.out, args.snapshot)

