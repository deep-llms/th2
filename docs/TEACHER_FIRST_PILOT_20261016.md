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

## First attempt and numerical-reference correction

Attempt a01 passed checkpoint/data/environment preflight, copied and verified
Accelerate, reclaimed exact known burns and verified all GPUs free. Its first
smoke stopped before fitting: BF16 FA4 vs SDPA loss gap0.00184155 exceeded the
newly introduced0.001 threshold; gradient relative L2 was0.010645. Burns were
automatically restored and verified at11:01:49UTC. No full fits ran.

Retry retains the same scientific scope, weights/data/branch initialization and
fresh outputs. It uses the existing trained-checkpoint numerical tolerances from
`scripts/check_proxy_trained_attention.py`: absolute loss0.01, gradient relative
L2 0.03. To investigate rather than merely waive the first failure, it adds an
independent full-FP32 math-SDPA forward/backward reference and requires BOTH BF16
paths to agree with it within those tolerances. Failed a01 evidence is retained.

Final paired validation is also repeated with dense SDPA for both fitted branches
and A. The teacher signal must meet the same scientific gate in BOTH backends;
the scientific effect threshold is not relaxed. All8-rank smokes must still pass.

## Completed result

Retry a02 completed all five jobs successfully at11:26:05UTC. Both8-rank smokes,
independent FP32 references, exact zero identity, frozen-weight verification and
save/reload checks passed. Largest gradient relative L2 against FP32 was1.2451%.
Verified communicating burns were restored11:27:15UTC; the11:27:17 live check
confirmed matching ownership, advancing progress (log5.03seconds old) and a
released GPU guard. No remaining training or student distillation is queued.

On the fixed512-row validation probe (these are **not full validation PPLs**):

| Method | FA4 LM loss | FA4 PPL |
|---|---:|---:|
| Frozen A | 3.60070445 | 36.624025 |
| Learned shallow-memory control | 3.60019471 | 36.605361 |
| Learned real-deep-memory teacher | 3.60018137 | 36.604873 |

Deep minus shallow: -0.00001333 nats/token, exploratory packed-row95% CI
[-0.00008960,+0.00005979]. One row half favors shallow. Dense SDPA independently
gives -0.00000109, CI[-0.00007650,+0.00007023], also effectively tied.
Both branches improve over A by about0.0005 nats/token, but there is no evidence
of an advantage specifically from real deep features. The preregistered gate
returns **stop_no_teacher_advantage**. Do not proceed to student distillation
or interpret this short, single-seed pilot as a universal impossibility result.

Actual fitting: shallow19.30seconds, deep21.07seconds. Successful numerical
gates, both fits and dual-backend validation together took285.55seconds (4m46s)
across eight GPUs, plus45.17seconds for a01's failed initial numerical smoke.
These times exclude preparation, burn handoffs and delayed controller reporting.

Exact remote summary bytes reconstructed locally and SHA256 verified:
`7d1bdccac1ee030b1850e0242d0687c80c9e3b2d80f813bb5b229adf520d4290`.
Results: [teacher_first_pilot_20261016.json](results/teacher_first_pilot_20261016.json).
Numerical/preflight/completion/live-burn receipts:
[teacher_first_completion_20261016.json](results/teacher_first_completion_20261016.json).
