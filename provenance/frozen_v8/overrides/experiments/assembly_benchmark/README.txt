FRAME SUPERVISORY BENCHMARK v8

Purpose
Compare Qwen/Qwen3.5-4B and gpt-6-astra choosing from identical existing skills.
The user authorized Codex-backed Astra in v4 instead of supplying an API key.
This is a comparison of deployed systems; it is not a pure model ablation.
This is NOT yet a code-as-policy/direct-action comparison, nor an RL experiment.
The API encodes a substantial task sequence and the skills solve the motor tasks.
A successful run alone does not establish that images were necessary to the model's
decisions. Raw image ablation would be a separate future experiment.

Audit
Geometry, masses, robot limits, friction, contact compliance, fastener parameters,
cameras and starting-state model arrays are compared against successful run83.
Five loose workpieces remain free. Native fingers have finite contact forces;
nails remain finite point springs with irreversible failure. No glue is modeled.
Physical validation is absent. Known dimensions, calibrated cameras, brown-material
segmentation, support height and no-slip propagation are strong perception priors.
The RGB estimator is unchanged and is NOT a learned VLA.

Benchmark-specific changes
New model-neutral runner; actual returned model/request metadata replace the old
presence-of-marker-file attribution. Strict action JSON has no execution capability.
The task inputs contain only serialized public observations and images.
The HTTP route supplies no tools. The Codex route disables shell, image-file tools,
integrations, browser/computer use, memories and delegation; any tool event rejects
the response. It uses a fresh public-only temporary directory and ephemeral session.
Desktop parent tool/permission environment overrides are not inherited.
Explicit park_grippers exposes run83's existing clearance step to BOTH models.
The reused live Executor otherwise retains its existing skill implementation.
Compressed streaming telemetry avoids accumulating huge JSON traces.
New acceptance gates require full release/flip retention and track airborne motion.
Unintended robot-workpiece contact classification covers the flip at 20 ms intervals;
the substep penetration gate covers the entire episode. This does not establish the
absence of earlier unintended contacts or sub-20 ms contact events.
No tuning of physical parameters, fasteners or geometry for either model.
The v1 scripted harness trial failed when the moving feeder occluded the frame and
overwrote observed frame estimates with missing detections. v2 freezes the last
visually verified frame estimates at feed start, an explicit stationary-fastened-frame
prior already used in the successful scripted pipeline. No ground-truth replacement
is used; unseen subsequent frame motion remains a limitation. The failure is retained.
Before model evaluation, the user authorized RunPod for inference, superseding the
earlier infrastructure constraint. v3 permits this and records the amendment; the
physics, skills, observations, test seeds and acceptance gates are unchanged from v2.
Historical v1/v2 bundles remain intact. Model weights and serving software/hardware
are pinned separately in deployment metadata before any scored episode.
v4 adds the user-requested Codex route. The common task instructions replace its
default coding instructions, but the CLI adds wrapper context; full neural prompts
are not identical. It attaches the same original images in camera0/2/3 order.
No CLI structured-output repair or constrained schema is used.
Codex's model identity is configured rather than independently returned as an
immutable backend snapshot. It has no documented generation-time output-token cap;
over-4096 reported-token responses fail a post-response gate. Qwen's cap is enforced
during generation. Request timeouts remain identical. Record this budget mismatch.
v5 reverses within-pair order before any held-out results: Astra first in pair1,
Qwen first in pair2, Astra first in pair3. This accommodates the Qwen host's direct
SSH outage without changing any physical or model-policy setting.
v6 follows an audited failure in the first v5 Astra pilot. At placement completion
the command target jumped6.67 mm, followed by drift while the model was reasoning.
The shared controller now preserves its previous settled command on this handoff.
The old pilot remains a failure; it is not rerun and relabeled a held-out success.
The new comparison uses untouched seeds22001..22003 after a paired development
regression with an explicit six-second decision delay. Physics parameters,
fastener strength, fixture geometry and acceptance thresholds are unchanged.
Before any new held-out trials, v7 uses Qwen's published thinking-mode sampler for
general tasks rather than the earlier coding-oriented configuration. The exact
parameters and primary source are recorded in protocol.json. No parameter sweep,
task fine-tuning, or adaptation to held-out Qwen results was performed.

v8 extends the existing two-current-image silhouette fit to released front/rear
rails after a delayed development test exposed single-view angle bias at a joint.
The surveyed planar support assumption remains explicit. Missing second-view
evidence rejects the estimate; no hidden state or target replacement is used.
No held-out v7 cases started; fresh paired seeds and all acceptance gates remain.

Interpretation
This measures supervisory decision quality given a strong shared controller.
Neither model is trained on this task. Model sizes, pretraining, hidden reasoning,
provider image processing and inference hardware are not matched. Same sensor bytes
and schemas do not imply identical neural visual tokenization.
Paired resets are identical; after models make different decisions or have different
latencies, their observations may diverge. Compare logged initial image hashes.
Three cases per model provide descriptive pilot evidence, not high reliability.
Older experiments and development seed3 must not count as held-out trials.

Timing
Each model call runs in a worker while physics steps. If local simulation falls
behind wall time, responses are still held until their measured inference delay has
elapsed in simulated time. This prevents discounting latency with a slow simulator.
Synchronous motion planning still stalls the integrator; stalls and wall lag are
reported. The current runner therefore does NOT establish executable real-time
robot control. Failed provider calls and invalid/truncated outputs are retained.
No automatic retries or model-specific output repair.

Freeze
  .venv-frame/bin/python -m experiments.assembly_benchmark.freeze OUTPUT_BUNDLE
  .venv-frame/bin/python -m experiments.assembly_benchmark.freeze OUTPUT_BUNDLE --verify

All source packages and both needed robot asset trees are copied and hashed.
Files are read-only and verified before each trial. Versions are pinned. Launch
from bundle cwd with the absolute path of the original Python interpreter and -B.
Put run outputs OUTSIDE the bundle. Do not edit a frozen bundle; create a new version
and invalidate paired comparisons if the runtime changes.

Run (from bundle cwd; placeholders require actual configured credentials/endpoints)
  /ABS/phyla/.venv-frame/bin/python -B -m experiments.assembly_benchmark.run \
    --bundle /ABS/BUNDLE --out /ABS/RUN --model qwen \
    --served-model Qwen/Qwen3.5-4B --backend chat \
    --base-url http://127.0.0.1:8000/v1 --revision ACTUAL_WEIGHT_COMMIT --seed 22001
  /ABS/phyla/.venv-frame/bin/python -B -m experiments.assembly_benchmark.run \
    --bundle /ABS/BUNDLE --out /ABS/RUN --model astra \
    --served-model gpt-6-astra --backend codex --base-url codex-local \
    --revision codex-cli-0.162.0-alpha.17.2:gpt-6-astra-mutable-alias --seed 22001

An alias is mutable: record that limitation if a provider has no immutable snapshot.
Record Qwen precision/quantization and hardware in deployment metadata. Do not
silently substitute a quantized/community model or a different parameter count.

Development transport/physics test (not actual model inference)
  .venv-frame/bin/python -m experiments.assembly_benchmark.mock_server --full
  Run the frozen runner using --development-smoke --harness-test --seed 3,
  --served-model HARNESS_TEST_ONLY --revision scripted-test --backend chat,
  --base-url http://127.0.0.1:18765/v1. For a brief stop-only smoke omit --full.
The result has model=null and infrastructure_only=true and is excluded from summary.

Tests
  .venv-frame/bin/python -m unittest experiments.assembly_benchmark.test_benchmark
  .venv-frame/bin/python -m unittest experiments.torso_assembly.test_flip \
    experiments.torso_assembly.test_fasteners experiments.torso_assembly.test_blind_pose \
    experiments.torso_assembly.test_multiview_pose experiments.torso_assembly.test_contracts

Report
  .venv-frame/bin/python -m experiments.assembly_benchmark.summarize OUTPUT_ROOT

Model access is required before calling this benchmark executed. No distillation,
RL or new robot embodiment is part of this stage.
