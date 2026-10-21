#!/usr/bin/env bash
set -euo pipefail
if [ "$(uname -s)" != Linux ] || [ "${PHYLA_CLOUD_RUN:-}" != 1 ]; then
  echo 'Run only on the authorized Linux cloud host with PHYLA_CLOUD_RUN=1.' >&2
  exit 1
fi
cd "$(dirname "$0")/../.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python3 -m venv .venv-camera
.venv-camera/bin/python -m pip install -r experiments/astra_camera/requirements.txt
mkdir -p outputs/frame_assembly/astra_camera
.venv-camera/bin/python -m pip freeze > outputs/frame_assembly/astra_camera/cloud_requirements.txt
.venv-camera/bin/python -u -m experiments.astra_camera.harness "$@"
