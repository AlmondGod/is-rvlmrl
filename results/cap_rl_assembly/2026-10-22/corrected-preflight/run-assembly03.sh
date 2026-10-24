#!/usr/bin/env bash
set -euo pipefail
cd /opt/cap-assembly/corrected-repo
export HF_HOME=/opt/cap-assembly/hf HF_HUB_ENABLE_HF_TRANSFER=0 MUJOCO_GL=egl CAP_RL_ASSEMBLY_CACHE=/opt/cap-assembly/assembly03-evidence
/opt/cap-assembly/venv/bin/python - <<'PY'
import json
assert json.load(open('/opt/cap-assembly/corrected-parallel-check/result.json'))['passes']
assert all(r['success'] for r in json.load(open('/opt/cap-assembly/corrected-controller-verification.json'))['results'])
PY
/opt/cap-assembly/venv/bin/python -u -m experiments.cap_rl.train train --task assembly_full --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --out /opt/cap-assembly/assembly03 --steps 32 --prompts-per-step 2 --group-size 4 --microbatch 1 --physics-workers 4 --max-tokens 512 --eval-every 8 --lr 0.0001 --support-profile contact_and_nails --controller-profile southwest_approach --checkpoint-every 1
