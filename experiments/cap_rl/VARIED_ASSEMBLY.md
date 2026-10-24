# Parallel RL with varied initial plank placements

This experiment continues Qwen3.5-4B CaP-style program GRPO from the previously
validation-selected strict-success adapter at step 2. The original model
revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Raw Qwen and that prior
adapter are comparison policies. No successful scripted program is supplied as
a training target. Programs still invoke the same RGB-feedback skill library.

Before training or inspecting new test results, the fresh `varied_v2` split is:
32 training seeds 51001–51032, eight validation seeds 52001–52008, and 24 held-out
seeds 53001–53024. The seed is hidden from the policy. Each side randomly swaps
its two feeder lanes; every plank gets independent X perturbation ±12 mm,
Y perturbation ±35 mm, and yaw ±0.06 rad (about 3.44°). These are the original
scene distribution and support geometry, with fresh layouts, not a changed
physics task. Unsuccessful layouts are retained.

The reward is strict physical success, 0 or 1. All released-frame geometry,
sheet stability, eight intact fasteners, airborne relative motion, contact,
numerical and five-second inverted-hold requirements stay unchanged. Dense
progress remains diagnostic only. A truncated response is unscorable and is
assigned zero before physical execution. Complete responses retain the actual
physical evaluator; there is no positive reward predictor or repaired program.

Initial training configuration: 16 rollout rounds, four prompts per round, four
samples per prompt, temperature 1.2, 512 completion tokens, learning rate
0.00005, KL coefficient 0.005, rank-16 all-linear LoRA, seed 42, validation
every four rounds, adapter snapshot every round. Four clipped GRPO optimizer
epochs reuse each physical rollout batch: 64 optimizer steps and 256 sampled
programs if all planned rounds complete. Old rollout and reference probabilities
remain frozen across these epochs. Temperature is applied to
both sampling and all old/current/reference log-probabilities. The unchanged
rollout policy computes its log-probabilities concurrently with physics; the
optimizer waits for every reward, so no stale-policy or off-policy update is
introduced. Extensions or changes may use training/validation evidence only.

Independent fresh physical replay is preselected for held-out seeds 53001,
53012 and 53024, regardless of success/failure, after checkpoint selection and
the primary evaluation. All test records also undergo certificate and model
regeneration checks.

Checkpoint selection uses validation strict success, with the initial adapter
eligible. The intended final target is at least 20/24 held-out strict successes;
it is a target, not a promised or measured outcome. Report raw, prior and selected
policies, paired wins/losses and uncertainty. Hold-out results must not guide
training or physics/controller changes. If the target is missed, report that
without removing cases or replacing the test set to conceal failure.

The validated hybrid backend uses GPU-batched Qwen generation/optimization,
GPU EGL camera rendering and independent CPU MuJoCo simulation processes.
OpenCV threads are bounded per worker to avoid oversubscription. Parallel
worker counts are chosen using development-prefix throughput and exact paired
state comparisons, not test success. Native GPU MuJoCo/Warp throughput is a
separate feasibility measurement until the RGB controller, custom finite nail
forces and acceptance measurements have a validated integration; kernel-only
speed must not be presented as end-to-end training speed.
