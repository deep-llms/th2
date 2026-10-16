# Teacher-first useful-memory gate

The latest proxy screen did not establish a useful downstream advantage. Before
training another predictor, test whether a *learned* deep-memory branch helps
the actual frozen LM more than the same branch reading shallow features.
This is the first prerequisite of the proposed teacher → contextual student
idea, not another full pretraining run. User authorized a cheap trial.

## Exact experiment

Start from the existing 12-layer A, seed42, checkpoint2500. Strict-load and
freeze every baseline parameter. Use one location only: input of block4, with
memory from output of block8 of the **unchanged frozen A trajectory**.
The memory therefore comes from a separate teacher computation; it is not the
deep state of the modified consumer trajectory or a recurrent fixed point.

Deep and shallow control use identical branch initialization (seed43):

```
source = original h8 for deep; original input-to-block4 for shallow
z_j = RMS(E RMS(source_j))                  # width128
q_t = Wq RMS(input-to-block4_t)
k_j = Wk z_j; v_j = Wv z_j                 # two heads, dimension64
delta_t = Wo Attention(q_t, k_{j<t}, v_{j<t})
h4'_t = input-to-block4_t + delta_t
```

Attention is strictly past and within the same packed document; document-start
messages are exactly zero. The small branch uses dense SDPA without additional
positional encoding. Existing backbone FA4, reset RoPE and document isolation
are unchanged. Wo starts at zero, giving exactly A before fitting. The source
features are detached but E, Q/K/V and Wo are learned through blocks4–12 and the
original LM vocabulary head, using ordinary next-token CE. No hidden matching
loss, independent vocabulary readout, proxy predictor or changed backbone.

Both fits use HF Trainer/Accelerate, eight GPUs, microbatch2, accumulation1,
sequence2048: 32,768 input tokens/update, 256 updates = 8,388,608 input tokens.
LR3e-4, 16 warmup steps then constant; AdamW, weight decay0.01, clip1, BF16;
seed/data seed42. Exact scored-token denominator across DDP ranks. These are
small adapter fits, not continuations of the original backbone optimizer.

Reuse train.py's existing tokenizer/packing/shuffle cache; verify its original
fingerprints. Freeze the first4096 shuffled training rows and512 evenly spaced
original validation rows into this pilot's output root. Same subsets and order
for both branches; no change to shared data or caches. No development tuning.
Validation rows are exploratory held-out data already used by earlier screens.

## Cost and acceptance

CPU tests check independent strict attention equations, no future/document leak,
zero identity, real frozen-tail gradient flow, unchanged weights, reload and HF
Trainer. Each real fit checks nonzero-branch FA4 versus SDPA loss/gradients, then
zero-initialized equivalence on real rows. Two-step eight-rank smokes precede
both256-step fits. Abort if first16 updates project a fit longer than10minutes;
each entire fit job has a hard15minute timeout. Failed jobs restore burns.

Evaluate A and each trained branch on the same512 rows. Report token-weighted
loss, paired packed-row bootstrap CI and both alternating row halves. This is
not a document-level significance test or a multi-seed claim.

Proceed-worthy teacher signal requires BOTH deep-minus-A and deep-minus-shallow
below -0.001 nats/token, upper exploratory95% CI below0, and improvement in both
halves. The threshold is frozen before seeing results. Failure stops this small
trial; it does not prove no teacher could work after more training. Success is
only a reason to consider student distillation, not evidence that the student
can reproduce the benefit or beat equal-compute A. Deep has extra source-pass
cost, which must be included in any later efficiency claim.

No student training, hyperparameter sweep, downstream evaluation or longer
training automatically follows. Save branch weights, source/data hashes, paired
row losses, timing, numerical checks and unchanged-backbone verification.

## Execution

`deep_kv/teacher_first.py` contains only the experimental module and minimal
HF loss/statistics adaptations. `scripts/teacher_first_pilot.py` freezes inputs,
builds the bounded jobs and validates results. Production A/P6/P7 code is unchanged.

Use `scripts/launch_teacher_first.sh FRESH_ROOT` under the existing tmux/runner
workflow. Preflight verifies completed prior evaluation, live communicating burn
ownership, source checkpoint hash, environment, data and Accelerate cache.
Existing `train_then_burn` rechecks and stops only verified burn workers, verifies
empty GPUs, executes the jobs, and restores verified burns on success/failure.
