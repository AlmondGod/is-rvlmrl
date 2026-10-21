"""Relocate a recorded trial into a new audit directory, then run frozen scoring.

Only file paths change. Original records are never edited. Raw arrays and traces
are symlinked read-only by convention; all audit writes go to the new directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    bundle, trial, out = map(lambda p: Path(p).resolve(), (args.bundle, args.trial, args.out))
    source = trial / "evaluator"
    required = ("raw.npz", "trace.jsonl.gz", "configuration.json", "result.json", "scene.xml")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        parser.error("Cannot rerun the full audit; missing original evidence: " + ", ".join(missing))
    config = json.loads((source / "configuration.json").read_text())
    metadata = json.loads((bundle / "freeze.json").read_text())
    if config["freeze"]["manifest_sha256"] != metadata["manifest_sha256"]:
        parser.error("Trial and bundle have different freezes; never silently substitute v9 for v8")
    scene = ET.fromstring((source / "scene.xml").read_text())
    changes = []
    for element in scene.iter():
        for key, value in list(element.attrib.items()):
            if "/third_party/" in value:
                relative = "third_party/" + value.split("/third_party/", 1)[1]
                replacement = bundle / relative
                if not replacement.exists():
                    parser.error("Missing asset: " + relative)
                element.set(key, str(replacement))
                changes.append({"attribute": key, "asset": relative})
    out.mkdir(parents=True, exist_ok=False)
    target = out / "evaluator"
    target.mkdir()
    for name in ("raw.npz", "trace.jsonl.gz"):
        (target / name).symlink_to(source / name)
    shutil.copyfile(source / "result.json", target / "result.json")
    config["arguments"]["bundle"] = str(bundle)
    config["arguments"]["out"] = str(out)
    (target / "configuration.json").write_text(json.dumps(config, indent=2))
    ET.ElementTree(scene).write(target / "scene.xml", encoding="unicode")
    (out / "relocation.json").write_text(json.dumps({
        "source_files_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                for name in required},
        "changes": changes, "numeric_state_or_model_parameters_changed": False,
        "purpose": "Path-only relocation for independent audit; not a new simulation trial"
    }, indent=2))
    subprocess.run([sys.executable, "-B", "-m", "experiments.assembly_benchmark.audit_saved", str(out)],
                   cwd=bundle, check=True)


if __name__ == "__main__":
    main()

