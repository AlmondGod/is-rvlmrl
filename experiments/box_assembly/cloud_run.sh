#!/usr/bin/env bash
set -euo pipefail
if [ "$(uname -s)" != Linux ] || [ "${PHYLA_CLOUD_RUN:-}" != 1 ]; then
  echo 'Run on the authorized Linux cloud host with PHYLA_CLOUD_RUN=1.' >&2
  exit 1
fi
cd "$(dirname "$0")/../.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python3 -m venv .venv-box
.venv-box/bin/python -m pip install -r experiments/astra_camera/requirements.txt
.venv-box/bin/python -u -m experiments.box_assembly.cloud_validate "$@"
