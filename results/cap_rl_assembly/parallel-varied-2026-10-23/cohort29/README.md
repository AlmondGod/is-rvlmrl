# Global cohort GRPO component

This component dispatches all groups from one unchanged policy without waiting
between generation batches, retains the real pending reward futures and frozen
old/reference probabilities, and waits for every measured reward before any
update. Each optimizer step accumulates gradients across the entire cohort.
It supports different prompt lengths without flattening or losing group layout
boundaries. The loss is the existing clipped GRPO objective; there is no reward
prediction, synthetic success or policy update during cohort generation.

Three small CPU tests compare the resulting parameters against an independent
per-sequence global mean, reject mixed policy versions/incomplete groups/nonbinary
rewards, and ensure dispatch does not wait for earlier rewards. These are
component tests, not a real Qwen or end-to-end RL benchmark. Online25 remains
unchanged. A subsequent experiment must freeze its rollout budget, initialization,
learning rate, validation schedule and selection rule before using this component.
