# P7 tuning variants

Implementation/preparation only. No new B200 job has been submitted and the
active exploration queue is unchanged. All cases below are fresh 2,500-step
12-layer runs using the existing `train.py` / HF Trainer / Accelerate path.
The complete list is **20 runs**, not a promise to finish within 12 hours.

## Targetgrad family

Parent: `P7-simple-targetgrad`, cosine auxiliary loss, maximum lambda 0.1,
four-block target, proxy sites 2/4/6/8, four selected query heads per site.
Each target is the current block's MLP output plus the next three MLP outputs,
followed by the existing running per-channel standardization and clipping.
Target statistics are detached; the standardized target has a live gradient.
The predictor input is detached and its output is detached before the memory
projections: LM gradients train the projections but not the predictor. Auxiliary
gradients train the predictor and reach the backbone through the target. That
target route can also reach memory projections through the backbone computation.

| Case label | Arm / change from parent |
|---|---|
| `tg-lambda003` | Parent with lambda 0.03 |
| `tg-lambda03` | Parent with lambda 0.3 |
| `tg-short` | `P7-simple-targetgrad-short`: target blocks l and l+1 |
| `tg-medium` | `P7-simple-targetgrad-medium`: target blocks l through l+2 |
| `tg-weighted` | `P7-simple-targetgrad-weighted`: coefficients 1.6/1.2/0.8/0.4 before standardization |
| `tg-huber` | `P7-simple-targetgrad-huber`: SmoothL1, beta=1, coordinate mean then global token/site mean; cosine remains a diagnostic |
| `tg-keyonly` | `P7-simple-targetgrad-keyonly`: proxy keys, native values |
| `tg-valueonly` | `P7-simple-targetgrad-valueonly`: native keys, proxy values |
| `tg-nonlinear` | `P7-simple-targetgrad-nonlinear`: separate residual transforms before K and V projections |
| `tg-sparse` | `P7-simple-targetgrad-sparse`: sites 2/6; same four-block targets |
| `tg-heads2` | `P7-simple-targetgrad-heads2`: last two query heads / one complete KV group |
| `tg-heads8` | `P7-simple-targetgrad-heads8`: last eight query heads / four complete KV groups |

All non-weight variants retain lambda 0.1. Shorter targets retain the parent's
sites; they do not introduce extra late-layer proxies. Native attention entries
remain present: these are proxy-enabled heads, not replacement of native heads.
The model must have two query heads per KV group and enough remaining native
groups; incompatible configurations fail before training.

Nonlinear transform: let p be the RMS-normalized prediction. Independently for
K and V, use `p + U SiLU(D p)` before the original projection, with bottleneck
width 32, no biases, random D and zero U. K still uses the existing head RMSNorm
and RoPE. This starts with the parent's exact forward computation. New modules
are initialized after all parent predictors/projections. Sparse arms preserve
the initialization at retained sites. Head-count variants preserve backbone and
predictor initialization; projection shapes and their random draws differ.

SmoothL1 uses the raw predictor output and standardized/clipped target; it is
not normalized MSE disguised as another cosine loss. Its numerical scale differs
from cosine, so lambda 0.1 is an initial comparison, not an equal-gradient claim.

## LM family and replication

These keep the LM parent's detached shallow input and detached target. LM loss
does train the predictor through its memory entries.

| Case label | Arm / change |
|---|---|
| `lm-lambda001` | `P7-simple-lm`, lambda 0.01 |
| `lm-lambda1` | `P7-simple-lm`, lambda 1.0 |
| `lm-sparse-noaux` | `P7-simple-lm-sparse-noaux`, sites 2/6 and lambda=0 |
| `lm-keyonly-noaux` | `P7-simple-lm-keyonly-noaux`, proxy keys/native values, lambda=0 |
| `lm-valueonly-noaux` | `P7-simple-lm-valueonly-noaux`, native keys/proxy values, lambda=0 |
| `lm-valueonly-gated` | `P7-simple-lm-valueonly-gated`, lambda 0.1, per-query-head value mixing |
| `a-seed2042` | A, training/data seed 2042 |
| `lm-noaux-seed2042` | Dense `P7-simple-lm-noaux`, training/data seed 2042, proxy seed 2043 |

Gated value mixing is `(1-g) Vnative + g Vproxy`, with a learned sigmoid gate
for each of the four selected query heads, initialized to 0.05. It uses native
attention probabilities, so there is no separate proxy-attention probability.
The existing mass metric records the effective value mixture weight, explicitly
identified in saved metadata. Per-head gates are also logged at normal training
log intervals and saved in checkpoints. Noaux arms reject a nonzero lambda;
target/cosine diagnostics remain available but create no auxiliary backward.

The existing parent already supplies both proxy K and V and separate linear
K/V projections. Those controls are not duplicated as new architectures.

## Throughput and correctness

- Value-only arms collapse identical keys: joint attention over `[K,K]` and
  `[V,Vproxy]` equals native attention over `(V+Vproxy)/2`. One ordinary attention
  call per decoder block replaces the extra selected-head attention call.
  This is real-arithmetic forward/VJP equivalence, not bitwise BF16 equivalence.
- Gated values similarly mix before attention. Independent query-head gates
  require expanded GQA K/V entries; they still use one native-length attention.
  This trades memory traffic for eliminating the second attention computation.
- Unused projections are removed; masks and document layouts are shared across
  sites; target collection reuses the existing forward's MLP outputs. Value-only
  mass diagnostics are analytic, avoiding the quadratic diagnostic softmax.
- Head-count support generalizes slicing and projection widths on both backends.
  Existing arm names, initialization, optimizer parameter names and algorithms
  retain their previous meaning. EOS packing and reset positions are unchanged.
- Existing validated compiled/LSE/fused-target options remain enabled only for
  the unmodified targetgrad lambda cases. New variants cannot accidentally call
  the fixed-four-block kernel with a different loss, span or head layout.

No measured B200 speedup is claimed for these new arms. CUDA numerical gates,
eight-rank smoke receipts, peak memory and actual training timing are still
required on B200 before a full run. The prepared numerical worker exercises
nonzero nonlinear transforms and nonuniform gates; value-only implementations
are compared to independent duplicate-entry or two-output-attention references.

## Reproducible execution

Definitions: `deep_kv/p7_variants.py`; model/loss implementation uses the existing
proxy modules. `scripts/p7_tuning_queue.py` prepares a sequential manifest:
numerical gates, all selected three-step distributed smokes and validators, a
shared-recipe/data gate, then full training/validation per case and an aggregate
summary/retention check. Each case has its own output directory. The full manifest
has 83 stages. Numerical probes cover 19 explicit proxy cases, preserving repeated
arms at different lambda/seed values, with each case's actual proxy settings and
optimization flags; A is covered by its distributed smoke. Recipe hashes are
recorded and checked before accepting the numerical summary.

```bash
python -m scripts.p7_tuning_queue \
  --recipe proxy_heads.12layer.b200.json \
  --rows /absolute/path/to/existing/rows.json \
  --output /absolute/path/to/new/jobs.json
```

This only writes files. `--cases tg-lambda003 tg-short tg-heads2` selects a smaller
subset in the fixed priority order. Generation refuses to overwrite a manifest
or configuration directory. The manifest uses eight GPUs per train/smoke,
microbatch 16, accumulation 4, sequence length 2048, 125 scheduler warmup steps,
250 auxiliary ramp steps, seed/data 42 and proxy seed 43 except the replication
pair. The preproduction gate compares each resolved smoke configuration against
its requested recipe, then checks identical training/evaluation fingerprints
within each seed and a common evaluation fingerprint across seeds. A different
seed intentionally changes the shuffled training fingerprint. It keeps and
verifies every checkpoint at 250-step intervals. No resume from a completed
2,500-step schedule is implied.

The generator supplies required `proxy_groups`, `proxy_lookahead` and
`proxy_loss_form` options for each fixed arm. Direct CLI calls must supply
`--proxy_groups 1` for heads2, `--proxy_groups 4` for heads8, and
`--proxy_loss_form smooth_l1` for huber. Mismatches fail rather than silently
changing an experiment. Lambda sweeps use `--proxy_lambda_max` on the parent.

Launch separately through the established supervisor, after verified predecessor
completion and burn handoff, Accelerate copy/verification and free-GPU checks.
The generator does not stop burns, launch a follow-up, or append downstream jobs.
Compare these full 2,500-step schedules with the existing full 2,500-step
controls, not intermediate checkpoints of the 7,500-step schedules. Downstream
evaluation and wall-clock comparisons remain necessary; PPL alone is not a win.

## Local evidence

CPU-only tests cover independent target sums, SmoothL1 and attention equations,
all gradient routes, FP32/BF16 checkpoint replay, SDPA versus a reference FA4
layout, document/future-token isolation, model budgets, recipe generation,
metadata rejection and checkpoint receipt validation. Actual `train.py` tests
exercise every new arm, exact uninterrupted-versus-resumed weights, evaluation
loading and reports. Two-rank Gloo and existing-arm regressions are tracked in
the current-task notes. These tests do not execute the CUDA FA4 kernel.

Final local result: **36 unique targeted test methods passed**, including all 14
new arms through real Trainer resume/reload and two-rank Gloo. See
[the validation record](results/p7_tuning_local_checks_20261015.json) for log
hashes, durations and review corrections. B200 numerical/timing checks remain
pending; no new queue was launched.

Follow-up review: **27 targeted test methods passed**, including a fresh rerun of
all 14 variants through exact Trainer resume/reload and two-rank Gloo. Fixed
per-case numerical recipe coverage and added shared-recipe/data and checkpoint
retention gates. Model computation was unchanged by this review. See the
[follow-up validation record](results/p7_tuning_review_20261015.json).
