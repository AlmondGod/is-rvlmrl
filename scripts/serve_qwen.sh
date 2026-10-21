#!/usr/bin/env bash
set -euo pipefail
# Same model/engine settings as the pilot, with machine-specific paths removed.
# Install requirements-serving.txt separately; this downloads ~9.3 GB of weights.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --served-model-name Qwen/Qwen3.5-4B \
  --host 127.0.0.1 --port "${QWEN_PORT:-8000}" \
  --dtype bfloat16 --max-model-len 16384 \
  --max-num-seqs 1 --gpu-memory-utilization 0.88 \
  --limit-mm-per-prompt '{"image":3}' \
  --reasoning-parser qwen3 --enforce-eager --seed 31019

