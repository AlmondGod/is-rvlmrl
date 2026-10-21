"""Report all real started model episodes, including errors; exclude mocks."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta

def interval(successes, trials):
    if trials == 0:
        return None
    return [0. if successes == 0 else float(beta.ppf(.025, successes, trials - successes + 1)),
            1. if successes == trials else float(beta.ppf(.975, successes + 1, trials - successes))]

def summarize(root):
    trials = []
    excluded = []
    for path in sorted(Path(root).rglob("evaluator/result.json")):
        value = json.loads(path.read_text())
        if value.get("infrastructure_only") or value.get("development_smoke"):
            excluded.append(str(path))
            continue
        trials.append({"path": str(path), **value})
    groups = {}
    for trial in trials:
        group = groups.setdefault(trial["model"], [])
        group.append(trial)
    models = {}
    for model, rows in groups.items():
        wins = sum(r["passes"] for r in rows)
        latencies = [c["latency_wall_s"] for r in rows for c in r["calls"]]
        models[model] = {"successes": wins, "started_trials": len(rows),
                         "exact_95_percent_interval": interval(wins, len(rows)),
                         "seeds": [r["seed_evaluator_only"] for r in rows],
                         "failure_classes": [r["failure_class"] for r in rows if not r["passes"]],
                         "latency_wall_s_p50_p95": np.quantile(latencies, [.5, .95]).tolist() if latencies else None}
    return {"scope": "Matched supervisory pilot; not code-policy/RL/motor-control evidence.",
            "models": models, "real_model_trials": len(trials), "excluded_development_or_mock": excluded,
            "no_model_results_yet": not trials, "physical_validation": False}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = json.dumps(summarize(args.root), indent=2)
    if args.output:
        Path(args.output).write_text(result)
    print(result)
