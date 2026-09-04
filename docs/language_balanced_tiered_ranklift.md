# Language-Balanced GroupReduce and Tiered RankLift

Status: implemented and CPU plus four-GPU CUDA/DDP verified; neither arm has
been trained yet.

## Purpose

The existing matched GroupReduce control ranks vocabulary rows by raw token
count. Because the training mixture is 85.7% English but evaluation gives each
of six languages equal weight, raw ranking allocates most high-rank rows to
English and places a disproportionate fraction of Chinese evaluation traffic
in the rank-64 tail.

These experiments separate two questions:

1. Does matching the static capacity ordering to the equal-language evaluation
   objective improve GroupReduce?
2. After fixing that ordering, does nonlinear classifier width help the large
   low-rank torso and tail groups at nearly the same whole-model size?

The comparison is intentionally paired. Both arms use the same fixed token
ordering from `resources/token_importance_langbalanced.npz`, the same four
group populations, exact input/output weight tying, and the same flat
151,936-way softmax.

## Language-balanced importance

For language `l` and token `w`, the stored importance is

```text
importance(w) = mean_l count_l(w) / sum_u count_l(u)
```

Thus each language contributes total mass `1/6` to the ordering regardless of
its training-corpus size. This is language-balanced *importance*, not balanced
training: the pretraining stream itself remains unchanged. Sorting is stable
high-to-low importance with token ID as the tie-breaker. Checkpoints persist
the resulting `group_ids`; evaluation and resume never depend on recomputing
the ordering from the external file.

## Arm 1: language-balanced GroupReduce

Experiment ID: `groupreduce_matched_lb_t4`

| Group | Population | Rank |
|---:|---:|---:|
| 0 | 2,048 | 1,024 |
| 1 | 6,144 | 512 |
| 2 | 24,576 | 192 |
| 3 | 119,168 | 64 |

The trainable interface has 19,423,232 parameters. This is exactly the
existing matched GroupReduce algebra and budget; only the static ordering
artifact changes from raw counts to equal-language importance. The arm is a
from-scratch tied adaptation, not a claim to reproduce GroupReduce's published
post-hoc SVD/refinement procedure.

Launcher: `scripts/train_groupreduce_matched_lb_tied.sh`.

## Arm 2: language-balanced Tiered RankLift

Experiment ID: `tiered_ranklift_lb_t4_c512`

Each group stores a private code `z_i` of width `c_g`. A lifted group computes

```text
u_i = RMSNorm(z_i)
g_i = SiLU(A_g u_i + a_g) * (B_g u_i + b_g)
f_i = concat(z_i, g_i)
e_i = f_i R_g^T
```

The exact tied classifier uses the same effective row:

```text
logit_i(h) = (h R_g) dot f_i
```

It therefore never constructs a dense `V x 1024` table. The two highest
capacity groups remain ordinary linear GroupReduce blocks. Nonlinear expansion
is applied only where classifier width is most constrained:

| Group | Population | Stored width `c` | Lift `q` | Classifier width `m=c+q` |
|---:|---:|---:|---:|---:|
| 0 | 2,048 | 1,024 | 0 | 1,024 |
| 1 | 6,144 | 512 | 0 | 512 |
| 2 | 24,576 | 192 | 320 | 512 |
| 3 | 119,168 | 64 | 192 | 256 |

The lift linears have learned biases; the final per-group projection remains
bias-free like the GroupReduce control. All four stored widths exactly match
the GroupReduce control, isolating the addition of nonlinear classifier
features in groups 2 and 3. Total interface size is 20,096,000 parameters:
672,768 more than language-balanced GroupReduce (3.46% of this interface and
about 0.15% of the full model).

Launcher: `scripts/train_tiered_ranklift_lb_tied.sh`.

## Interpretation

The first arm measures the effect of ordering alone against the existing raw-
frequency GroupReduce checkpoint. The second adds nonlinear width while
preserving every stored rank of the language-balanced GroupReduce control.
This is the cleanest first efficacy test: it avoids weakening group 1 merely
to satisfy a tiny whole-model parameter delta. If Tiered RankLift wins, compare
against a linear GroupReduce control at approximately 20.096M interface
parameters and optionally run the exact-budget Tiered 416 configuration before
making a strict equal-interface-budget mechanism claim.

A failure of the language-balanced control means the Chinese diagnosis was
not solved by ordering alone. A failure of Tiered RankLift against that control
means nonlinear classifier widening does not improve this tiered scaffold; do
not rescue it with a slower mixture head.

## Verification completed

- Exact tied-head values and gradients match a materialized effective table in
  float64.
- Zero-lift groups are algebraically identical to GroupReduce blocks.
- BF16 forward/backward connects every parameter, including when input IDs
  touch only one group (`ddp_find_unused_parameters=false` remains valid).
- Static non-contiguous memberships, offsets, inverse ordering, strict state
  loading, configless checkpoint reconstruction, and tiny-Qwen logits pass.
- Production parameter counts are asserted in tests.
- Both launchers pass `bash -n`; the new entry was appended so historical
  experiment indices remain stable.
- A production-shape BF16 forward/backward/AdamW step passed on four A100s for
  both arms with `find_unused_parameters=false`; every parameter had a finite
  gradient and the NCCL all-reduce marker was correct. The report is
  `temp/tiered_ranklift_gpu_smoke.json`. The 512-width production setting is
  revalidated before launch. All GPUs were free afterward.

CPU test:

```bash
pytest -q compositional/test_tiered_ranklift.py
```

Production CUDA/DDP interface smoke for the two new arms:

```bash
python scripts/smoke_final_compressed_interfaces_gpu.py \
  --arms groupreduce_matched_lb_t4,tiered_ranklift_lb_t4_c512 \
  --output temp/tiered_ranklift_gpu_smoke.json
```

The local four-A100 smoke passed. Repeating the short smoke on the B200 node is
still advisable as a hardware-specific preflight before a 10k launch.
