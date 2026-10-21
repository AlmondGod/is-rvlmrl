# Reproduction and evidence replay

Commands below run from the repository root unless explicitly changing directory.
Use fresh output directories. Do not overwrite recorded results or edit a frozen
bundle in place. A different source/dependency/CLI configuration requires a new
freeze and cannot silently join the historical pilot.

## Install and verify

Create a Python 3.12.9 environment and install `requirements.txt` as in the root
README. The simulator was run on macOS ARM without a CUDA GPU. Linux rendering
requires a functioning EGL/OpenGL environment; setting `MUJOCO_GL=egl` only
works when an EGL driver is actually installed.

```bash
python scripts/restore_frozen.py --out runtime/frozen_v9
python -m experiments.assembly_benchmark.freeze runtime/frozen_v9 --verify
```

Restoration copies only hash-verified files from the Git checkout and adds the
original manifest and runtime metadata. It must reproduce manifest
`e3e7642677e25162b853f1a5d552f89810f5ab281aecf499c78d81508deba952`.
This is a content check; the second command also checks Python/package versions.
No model weights or credentials are downloaded by these commands.

## Quick model-free smoke

In one terminal, activate the same virtual environment and start the deliberately
scripted server:

```bash
python -m experiments.assembly_benchmark.mock_server --port 18765
```

In another terminal, with the environment active, set absolute paths before
changing directory:

```bash
REPO="$PWD"
BUNDLE="$REPO/runtime/frozen_v9"
cd "$BUNDLE"
python -B -m experiments.assembly_benchmark.run \
  --bundle "$BUNDLE" --out "$REPO/outputs/smoke" \
  --model qwen --served-model HARNESS_TEST_ONLY --backend chat \
  --base-url http://127.0.0.1:18765/v1 --revision scripted-test \
  --seed 3 --development-smoke --harness-test --seconds 5
```

This returns a scripted `stop`. Expect **task acceptance false**,
`infrastructure_only=true` and `model=null`. This verifies the complete
observation/request/logging path; it is not an assembly success or model test.
Stop the mock server when done.

## Full scripted development demonstration

Restart the mock server with `--full --delay 6`. Run the same command with a new
output directory and omit `--seconds 5`. This exercises the full assembly and
flip sequence while physics continues through six-second mock inference delays.
The stored v8 development run passed at roughly 635 simulated seconds and 1,006
wall seconds on the recorded Mac. Runtime and outcome on other systems are not
guaranteed. The output remains explicitly a scripted development result.

v9's physics/controller bytes match that v8 regression. To reconstruct its exact
historical source, use `--snapshot v8 --out runtime/frozen_v8`; only the four
changed source files are stored as overrides, not a duplicate asset bundle.

## Actual model inference

On a separate Linux inference host, install `requirements-serving.txt`, then run
`bash scripts/serve_qwen.sh`. It uses original BF16 Qwen3.5-4B weights at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It downloads about 9.3 GB of weights.
The historical host had one RTX 4000 Ada with 20 GB VRAM. Keep the server bound to
loopback; establish your own authenticated access if simulation runs elsewhere.
No particular cloud provider or preexisting SSH host is required.

From a frozen bundle, use:

```bash
python -B -m experiments.assembly_benchmark.run \
  --bundle /ABS/runtime/frozen_v9 --out /ABS/outputs/qwen_development \
  --model qwen --served-model Qwen/Qwen3.5-4B --backend chat \
  --base-url http://127.0.0.1:8000/v1 \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --seed 3 --development-smoke
```

Use seed 3 for development. Seeds 23001–23003 are **already exposed**; replaying
them is a reproduction, not a fresh held-out test. Freeze a new distribution and
budget before a new scientific evaluation. The original Qwen arrangement used
an SSH HTTP relay with transport/setup time included in the timeout. Connecting
directly reproduces the interface and engine settings, not that network path.
Machine-specific relay credentials/endpoints are intentionally not included.

The historical Astra route uses `--model astra --served-model gpt-6-astra
--backend codex --base-url codex-local` and the revision label
`codex-cli-0.162.0-alpha.17.2:gpt-6-astra-mutable-alias`. It requires an existing
authorized Codex login and the exact CLI binary checked by `codex_provider.py`.
The path is the original macOS app path, and the binary hash is pinned. It will
not work with arbitrary CLI versions, Linux binaries or accounts lacking the
model. Changing those requires a new freeze, documented wrapper validation and
a fresh comparison. Never copy login files or API keys into this repository.

The Qwen route caps generation at 4,096 tokens; Codex has only a post-response
4,096 reported-token gate. Both have a 120-second request deadline. The configured
Codex alias is not an independently verified immutable backend model snapshot.

## Read and audit the saved evidence

Compact records, all saved decision images/packets, and responses are in Git.
Download the larger raw arrays, remaining traces and continuous videos from the
[snapshot release](https://github.com/AlmondGod/is-rvlmrl/releases/tag/research-snapshot-2026-10-21).
Verify archive hashes against `provenance/release_artifacts.json`, then extract
the evidence archive into a new directory. It contains `results/` paths.
File-level hashes and omissions are in `provenance/evidence_inventory.json`.

Post-hoc summaries use only compact records and do not drive the policy:

```bash
python analysis/analyze_cohort.py results/pilot_v9
```

This summarizes **available completed records**, so read it together with
`results/pilot_v9/cohort_status.json`. It refuses to pool different freezes.
Neither an empty directory nor the planned denominator is a completed outcome.

For a trace that is present, reconstruct the bundle and run:

```bash
python scripts/replay_audit.py \
  --bundle /ABS/runtime/frozen_v9 \
  --trial /ABS/extracted/results/pilot_v9/qwen_23001 \
  --out /ABS/outputs/qwen_23001_reaudit
```

This relocates asset paths and configuration paths into a new directory and
invokes the frozen auditor. Numeric states and model parameters are unchanged;
the originals are preserved. The auditor writes new consistency results to the
new directory. Use the v8 bundle for `results/development/delayed_full_v8`.

For Astra 23001/23002, the detailed traces were already missing at packaging, so
the full auditor must refuse to run. Prior audit reports are retained, but this
snapshot cannot independently regenerate them. Do not synthesize missing traces
from raw states and present them as recorded force/contact evidence.

## Legacy entry points

The seven source packages are deliberately unchanged to preserve the runtime
manifest. Some contain older scripts outside the active benchmark. In particular,
`experiments.wall_panel.test_contracts` initializes from an omitted historical
frame checkpoint; Wuji-hand scripts need assets/packages outside this snapshot;
rail-RL scripts require their separate dependencies. The documented root test
command avoids those unrelated entry points. The current torso environment does
not initialize workpieces from that old assembled-frame checkpoint.

No new training, long evaluation campaign or model queries are required to inspect
this snapshot. Packaging validation uses the core tests, exact restoration,
model-free smoke and a saved-record audit.
