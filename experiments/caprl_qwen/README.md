# Qwen CaP-RL MuJoCo pilot

This is an isolated proof-of-concept, not the existing frame-assembly benchmark.
A real Qwen2.5-Coder-0.5B-Instruct checkpoint generates Python push commands.
A strict AST whitelist validates code; every accepted command produces MuJoCo
contact dynamics and is rewarded based on measured cube-to-goal distance.
Optimization is LoRA SFT followed by on-policy group-normalized REINFORCE
(GRPO-style, without clipping). Same held-out seeds at each stage.

Run from repository root on a CUDA machine:

    python -m venv .venv-caprl
    source .venv-caprl/bin/activate
    pip install -r experiments/caprl_qwen/requirements.txt
    python -m experiments.caprl_qwen.environment
    python -m experiments.caprl_qwen.train

Check runs/caprl_qwen/baseline.json, sft.json, rl.json, and adapter.
No measured training gains are claimed until this GPU run finishes.

Scope limitations: only one/two cardinal push primitives, privileged XY
observations, deterministic controller pre-positioning. Not a visual VLM,
full general-purpose robot-control code, or CaP-X reproduction.
