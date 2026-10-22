"""Generate and execute a program with a saved adapter for a user-chosen layout."""
import argparse
import json
from pathlib import Path

import torch

from .task import Layout
from .train import load_model, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--gripper", nargs=2, required=True, type=int)
    parser.add_argument("--member", nargs=2, required=True, type=int)
    parser.add_argument("--target", nargs=2, required=True, type=int)
    args = parser.parse_args()
    points = [tuple(p) for p in (args.gripper, args.member, args.target)]
    if len(set(points)) != 3 or any(abs(x)>2 for point in points for x in point):
        parser.error("Choose three distinct cells with each coordinate in [-2, 2].")
    metadata = json.loads((args.run / "metadata.json").read_text())
    if metadata.get("task", "pick_place") != "pick_place":
        parser.error("This coordinate demo supports only the single-object task.")
    torch.set_num_threads(4)
    model, tokenizer = load_model(metadata["model"], metadata["model_revision"],
                                  str(args.run / "best_adapter"))
    layout = Layout(*points)
    with model.disable_adapter():
        _, _, _, before, before_results = rollout(model, tokenizer, [layout], metadata["max_tokens"], False)
    _, _, _, after, after_results = rollout(model, tokenizer, [layout], metadata["max_tokens"], False)
    print(json.dumps({"layout_id": layout.key,
                      "base": {"program": before[0], **before_results[0]},
                      "trained": {"program": after[0], **after_results[0]}}, indent=2))


if __name__ == "__main__":
    main()
