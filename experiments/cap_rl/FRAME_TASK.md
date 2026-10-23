# Qwen3.5 rail placement experiment

**Known frozen-reward loophole:** `frame01` can reward sliding the rail to the target and then lifting there. Preserve that source and its rewards for reproducibility. Report ordered pickup-and-carry success only after applying `python -m experiments.cap_rl.frame_order RUN --out RUN/ordered_test_comparison.json`. The additional check was fixed after training and before inspecting held-out results; it requires a centrally grasped rail at the requested clearance within 12 mm of its original center, followed by a later call boundary at clearance within 8 mm of the target. It is an evaluation check, not a claim that the original training reward enforced this order.


This is an isolated frame-assembly **pickup, transfer, placement, release and withdrawal** task. One native seven-joint xArm7 and its actuated two-finger gripper manipulate a free wooden rail under gravity in MuJoCo 3.3.7. It extends the earlier lift task by requiring a new target position and unsupported final placement. It does not assemble a joint, drive nails, use vision, or solve the full three-arm benchmark.

The rail is 0.038 × 0.9144 × 0.064 m, density 600 kg/m³. The existing `robot_lift.py` physics/controller is unchanged. No wood weld, mocap, external force or post-reset pose write is used. The native gripper's three linkage equalities remain.

## Layouts and API

Seed 20261027 generates 512 training, 64 validation and 128 test layouts. Full layout tuples are disjoint; coordinate values recur between splits. Gripper XY starts in integer centimeters within [-3,3]², rail and target centers within [-4,4]². Rail-to-target displacement length is 3–7 cm. Required minimum lift clearance is 6, 8 or 10 cm. Orientation starts fixed and must be retained.

The text prompt gives exact privileged coordinates and API semantics. It asks the model to plan the sequence, without supplying an example program or explicit argument formulas. The safe AST interpreter permits 1–10 literal API calls, never executes arbitrary Python, and rejects variables, imports, expressions and loops:

- `move(dx, dy)`: relative XY centimeter displacement, integers [-8,8].
- `lower()`: descend to calibrated TCP height 0.018 m, hold 0.3 s.
- `close()`: ramp native gripper command to 180 and wait 2 s.
- `lift(h)`: relative vertical centimeters, integer [1,14], hold 1 s.
- `open()`: ramp native gripper open and wait 2 s.
- `retreat()`: raise TCP to 0.20 m, hold 1 s.

The model selects high-level code and parameters; a supplied 50 Hz IK/joint controller supplies motor commands. No perception or low-level motor policy is learned.

## Reward and strict success

After each attempted call and each final hold tick, track maxima of:

1. **0.10 alignment**: `max(0, 1 - XY TCP-to-wood-center error / 0.06)`.
2. **0.15 contact**: finger normal force >5 N and central XY error ≤12 mm.
3. **0.15 elevation**: height gain / requested clearance, clipped [0,1], while centrally grasped.
4. **0.15 transport**: `max(0, 1 - rail-to-target XY distance / 0.08)`, only while centrally grasped at requested clearance (8 mm tolerance).
5. **0.10 placement**: `max(0, 1 - rail-to-target XY distance / 0.04)`, only after airborne arrival within 8 mm, then grounded, correctly oriented, with fingers open and no contact.
6. **0.35 strict success**.

Success requires an observed central grasp and lift to at least the requested clearance minus 8 mm, followed by airborne arrival within 8 mm of the target. During every tick of the final 0.5 s, require XY target error ≤8 mm, rail center height within 3 mm of 0.032 m, tilt and absolute yaw ≤0.05 rad, linear speed ≤0.01 m/s, angular speed ≤0.05 rad/s, gripper command zero, finger contact <1 N, and TCP at least 0.10 m above the rail center. Sliding without any lift, carrying without release, or releasing without withdrawing cannot pass. However, the original implementation allows sliding first and lifting at the target: the two stage flags can become true at the same call boundary. The separate pickup-first verification above closes this gap for reported ordered-carry results.

All episodes require peak penetration <3 mm, no MuJoCo warnings, and no runtime error for success. Unsafe physics zeros the full reward. Parser failures and incomplete generations score zero; runtime errors retain earned prefix reward but cannot succeed. Physical tolerance means success is not equivalent to exact canonical code.

## Feasibility and evaluation protocol

Before any model training, a scripted controller passed all 512 training layouts. This oracle is used only for simulator feasibility and diagnostics, never as model training data. Negative tests cover no grasp, inadequate lift, wrong target, no release, no retreat and unsafe Python.

The intended model is `Qwen/Qwen3.5-4B`, pinned to `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, initialized from pretrained weights, with thinking disabled for short executable programs. Use text-only causal language model weights, rank-16 all-linear LoRA, online group-relative policy optimization and KL to the immutable original model. Select a checkpoint on validation before the paired held-out test. Record the actual runtime, steps and final results separately; this specification makes no training-outcome claim.
