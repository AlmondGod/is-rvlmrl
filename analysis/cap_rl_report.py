"""Plot saved CAP RL results: pip install matplotlib==3.10.7."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.directory/"metadata.json").read_text())
    ordered = metadata.get("task") == "ordered_two"
    physics = metadata.get("task") == "mujoco_docking"
    placement = metadata.get("task") == "frame_place"
    robot = metadata.get("task") in ("robot_lift", "frame_place")
    report = json.loads((args.directory/"test_comparison.json").read_text())
    selection = json.loads((args.directory/"selection.json").read_text())
    history = [json.loads(line) for line in (args.directory/"training.jsonl").read_text().splitlines()]
    validation = [(0, json.loads((args.directory/"validation_baseline.json").read_text())["metrics"])]
    for path in sorted(args.directory.glob("validation_[0-9]*.json")):
        validation.append((int(path.stem.split("_")[1]), json.loads(path.read_text())["metrics"]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [1.55, 1]})
    steps = [row["step"] for row in history]
    moving = [100*sum(row["success_rate"] for row in history[max(0,i-9):i+1])/min(i+1,10)
              for i in range(len(history))]
    left.plot(steps, moving, color="#8ba9bb", linewidth=1.5, label="Training samples (10-update mean)")
    left.plot([s for s,_ in validation], [100*r["success_rate"] for _,r in validation],
              "o-", color="#12664f", linewidth=2.5, label="Greedy validation · 64 layouts")
    chosen = next((s,r) for s,r in validation if s == selection["best_step"])
    left.scatter([chosen[0]], [100*chosen[1]["success_rate"]], marker="*", s=180,
                  color="#dc8b23", zorder=4, label="Selected checkpoint")
    left.set(xlabel="GRPO update", ylabel="Strict task success (%)", ylim=(0, 105 if robot else 100),
             title="Learning curve")
    left.grid(axis="y", alpha=.16)
    left.legend(loc="lower right" if robot else "upper left", frameon=False, fontsize=8)
    values = [report[k]["success_rate"]*100 for k in ("baseline", "trained")]
    labels = (["One-object RL", "MuJoCo RL"] if physics and metadata.get("init_adapter") else
              ["One-object RL", "Two-object RL"] if ordered else ["Base model", "After RL"])
    bars = right.bar(labels, values, color=["#9aa3ac", "#12664f"], width=.58)
    for bar, key in zip(bars, ("baseline", "trained")):
        result = report[key]
        right.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                   f'{result["success_rate"]:.1%}\n{result["successes"]}/{result["n"]}',
                   ha="center", va="bottom", fontsize=11)
    right.set(ylim=(0, 112), ylabel="Strict task success (%)", title="Untouched test layouts")
    right.set_yticks([0,25,50,75,100])
    right.grid(axis="y", alpha=.16)
    right.set_axisbelow(True)
    title = ("xArm7: place and release a frame rail" if placement else "xArm7: grasp and lift a frame rail" if robot else "MuJoCo: two-station inertial docking" if physics else
             "ordered two-object delivery" if ordered else "discrete pick-and-place")
    fig.suptitle("Code-as-policy RL: " + title, x=.07, ha="left", fontsize=16, fontweight="bold")
    lo, hi = report["paired_bootstrap_95_ci"]
    fig.text(.07,.88,metadata["model"].split("/")[-1]+" · one training seed · same prompts and greedy decoding before/after", fontsize=9, color="#4c5660")
    limitation = ('Articulated arm and contact grasp; privileged coordinates and fixed motor skills; one seed, simulation only.' if robot else 'MuJoCo force-controlled planar slider; known dynamics, one seed, no robotic assembly or real-world transfer claim.' if physics else
                  'Symbolic coordinate task; no claim of MuJoCo assembly performance or physical transfer.')
    fig.text(.07,.03,f'Test gain: {report["delta_success_rate"]*100:.1f} percentage points (paired bootstrap 95% CI: {lo*100:.1f}–{hi*100:.1f}).\n{limitation}',fontsize=9,color="#4c5660")
    fig.subplots_adjust(left=.07, right=.97, bottom=.23, top=.77, wspace=.30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor="white")


if __name__ == "__main__":
    main()
