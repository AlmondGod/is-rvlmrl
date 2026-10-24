# Illustrative validation episode after eight GRPO updates

This is seed 52001, one validation win, not a held-out result. The baseline
scores 4/8 overall; checkpoint 2 scores 3/8, with one paired win and two losses.
It is unselected. The video replays certified recorded states at 20x speed;
it does not rerun dynamics. Source programs, outcomes and certificates are
included here. The final test split remains untouched.

The before program positions members in order 0,1,2,3. The after program
uses 0,1,3,2. Both invoke the same 22 supplied feedback-controlled skills,
including all eight nails and the flip. The policy is Qwen3.5-4B with initial
RGB-derived text and constrained code generation, trained by GRPO.
