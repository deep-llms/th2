# P6/P7 speed: remaining safe optimizations

Date: 2026-10-08. Reviewed at commit `6f7b77d`, after the execution
optimization in `PROXY_OPTIMIZATION_20261008.md`. Code reviewed:
`deep_kv/proxy.py`, `deep_kv/proxy_memory.py`, `deep_kv/proxy_estimators.py`,
`deep_kv/proxy_training.py`, `deep_kv/fa4.py`. No code was changed and no GPU work
was run for this review.

**Scope.** Only changes that leave each arm's definition intact: targets, loss
weights, gradient routing, initialization, data order, normalization rule and
attention semantics. Each item is classed as:
- **Exact:** bitwise-identical outputs and gradients. Verify with `rtol=atol=0`.
- **Equivalent:** same mathematics, different floating-point operation order.
  Verify with tolerances, as in the existing routed/reference CUDA checks. Apply
  only to fresh runs, never in the middle of a run or on resume.

## 0. Measure the current code first

The B200 times in `PROXY_RESULTS_SUMMARY.md` (A 88.3 min, P6-iso +9.3%,
P7-simple +15.5%, P7 about +25.6%) predate commit `88dd927`. No component
profile exists for P6/P7-simple; the latest is P7-kq on FA4
(`artifacts/p7-review-20261006/.../fa4/seed-42/P7-kq/component-profile.json`).

Before any change, run the existing 25-step, 8-GPU smoke with
`scripts/profile_proxy_training.py` for A, P6-iso, P7-simple and P7 on the current
code. Use the result as the baseline for every item below.

**Reference profile, P7-kq, FA4.** Exclusive kernel milliseconds per update on
rank 0. These are kernel sums, not critical-path wall time.

| Component | Forward | Backward |
|---|---:|---:|
| Proxy-layer attention (native and proxy calls) | 224 | 420 |
| Relational KL | 61 | 92 |
| Cosine loss | 24 | 51 |
| Estimator (with convolution) | 20 | 45 |
| Proxy K/V projections | 33 | 25 |
| Target normalization | 23 | — |
| Target/mask/statistics setup (extra vs A) | +14 | — |

The proxy layers' attention replaced about 526 ms of A's decoder time. The
duplicated-query attention therefore costs roughly 120 ms extra per update.
Relational, cosine, estimator, projection and target work account for most of
the rest of the ~540 ms gap. The P4-iso-4h profile shows the same pattern for
the tokenwise arms: the target, normalization and cosine path (~136 ms)
outweighs the estimator (~8 ms).

## 1. P7 family

### 1.1 Replace duplicated-query FA4 attention with an LSE merge (largest)

**Applies to:** every P7 arm on FA4. **Class:** equivalent.

**Current.** `flash_joint_attention` interleaves `[native_j, proxy_j]` and
duplicates every proxy query. Each of the 4 proxy heads then runs a causal
problem of length 2T, about 2T² work, and half of the outputs are discarded. It
also materializes stacked 2T key/value copies.

**Proposal.** Keep one softmax over both entry sets by merging two partial
softmaxes, the standard flash-decoding merge:
1. Call `flash_attn_varlen_func(..., return_lse=True)` once for all 16 heads with
   native K/V. This is A's own call, so the extra slicing of `q`/`k`/`v` into 12 and
   4 heads also goes away.
2. Call it once more for the 4 proxy heads against `[k^p, v^p]`, with the same
   causal varlen layout.
3. Merge the proxy heads in fp32:
   `m = max(lse₁, lse₂)`, `w_i = exp(lse_i − m)`,
   `out = (w₁·o₁ + w₂·o₂)/(w₁ + w₂)`.
   Heads 0–11 keep `o₁` unchanged.

Both calls have equal-length, causal, same-document visibility, which is
exactly the problem already validated. The merged result is the same joint
softmax, and the proxy-head work drops from about 2T² to T². The pinned
interface supports this for training: `FlashAttnVarlenFunc.backward(ctx, dout, dlse)`
consumes `dlse` when `return_lse=True`. In the saved b32 interface
(`temp/fa4-b32-interface.py`, lines 3511–3631), the only SM100 exclusion is
head dimension 256; Qwen3 uses 128.

**Estimate:** roughly 100–150 ms per update for P7 (≈4–6% of A's update time).
Unmeasured.

**Validation:**
- Confirm `return_lse`/`dlse` in the pinned **4.0.0b33** source, not only b32.
- CPU oracle test against the current interleaved path and SDPA `[M | M]`:
  outputs, LSE merge, and gradients for q, k, v, k^p, v^p.
- CUDA routed/reference check with the existing 1% gradient bound, and the
  cross-document leakage test.
- Proxy-masked evaluation, `gates_disabled`, must still use the plain native call.

### 1.2 Convolution without padding copies

**Applies to:** P7, P7-kq, P7-ems, P7-mlp (not P7-simple). **Class:** exact.

**Current.** `MemoryHead.estimate` multiplies each lag by a boolean mask, which
is converted to the activation dtype on every call. It then pads to full length
and adds, creating three temporary full-size tensors per lag.

**Proposal:**
- Store the masks once per `MemoryPlan` in the activation dtype.
- Accumulate lag `l` into the slice `c[:, l:]`, without `F.pad`.

Multiplying by 1.0/0.0 instead of True/False, and skipping the addition of exact
zeros, gives bitwise-identical values. Autograd is unaffected, because the
backward of `x * conv[0]` does not need `c`.

**Estimate:** part of the 65 ms estimator forward+backward.

**Validation:** existing convolution causality/reset tests, plus an
`rtol=atol=0` comparison against the current code.

### 1.3 Fuse the relational KL

**Applies to:** P7, P7-kq, P7-ems, P7-mlp. **Class:** equivalent.

`relational_losses` runs separate kernels for each masked fill, log-softmax,
exp, difference and sum on `[B, 256, T]` fp32 tensors, in forward and backward.
Keep both bmm calls in fp32, and fuse the elementwise and reduction tail:
- compile only this function, or
- use a custom autograd function that computes the KL gradient
  `(P^ŷ − P^tgt)/τ` directly.

**Estimate:** a fraction of the 153 ms relational forward+backward. The fp32
bmm itself stays.

## 2. Target, normalization and cosine path (P4/P6 family, P7-simple, P7)

### 2.1 Count clipped entries only on logging steps

**Class:** exact for training state.

`record()` computes `(|standardized| > clip).sum()` on every training microbatch
(`return_clipped=collect_target_statistics`), and these counts are all-reduced.
They feed only the logged `clip_fraction`. Passing the trainer's existing
`diagnostic` flag skips one full read of each target tensor on 9 of every 10
steps. Normalization buffers, losses and gradients are unchanged.

### 2.2 Fuse moments and normalization per target

**Class:** equivalent.

For each target window, `record()` makes separate passes for:
- `raw − μ`;
- the sum;
- the sum of squares;
- the scale multiply;
- the clamp;
- (with 2.1 off) the clip count.

One fused function (compiled, `dynamic=False`) produces the sums and the
normalized target in about two passes. The running-statistics update rule is
unchanged.

### 2.3 Window accumulation without reallocation

**Class:** exact.

`windows[p] = windows[p] + value` allocates a new fp32 `[B, T, d]` tensor for
every contribution. In-place `add_` gives identical values. The first
contribution is shared between two overlapping windows, so the window must own
its storage before the first in-place add: clone it, or start from the first
sum. The P6-iso-weighted coefficients are unchanged.

### 2.4 Fused cosine loss

**Class:** equivalent.

`F.cosine_similarity` on fp32 copies of `[B, T, 1024]` costs about 68–75 ms
forward+backward per update across 12 layers. The target is detached, so a
custom function can:
- normalize the target once;
- compute `cos = (p·t̂)/max(‖p‖, ε)` in one pass;
- return `∇p = (t̂ − cos·p̂)/‖p‖` in backward.

It must reproduce `F.cosine_similarity`'s epsilon semantics. Alternatively,
compile the existing `cosine_loss`.

### 2.5 P6 injection chain

**Class:** equivalent.

`normalize_code`, the α multiply, the detached residual RMS, the scale and the
residual add are separate fp32 elementwise kernels per proxy layer. Fusing them
(one compiled function) leaves the second `input_layernorm` call unchanged.

## 3. `torch.compile` precautions

Items 1.3, 2.2, 2.4 and 2.5 are easiest through `torch.compile` on **functions,
not modules**, as the existing `compile_estimator` option already does. That
option is currently off for P6 and rejected for P7, and it has never been
validated on CUDA. Before enabling it:
- Use `dynamic=False`, and confirm that no recompilation occurs. Training shapes
  are fixed, but a short final evaluation batch changes the shape.
- Check compatibility with activation checkpointing, DDP and the custom
  autograd functions. The recipe's checkpointing is currently off.
- Record the compile flag in the recipe (it already is), and compare peak memory.

## 4. Excluded: not safe without a specification change

- TF32 or BF16 for relational logits, cosine, target sums or statistics. The
  specifications require fp32 there.
- Fewer than 256 sampled relational queries, or computing auxiliary losses or
  statistics on a subset of steps.
- Removing the `isfinite` loss check or other host synchronizations. These also
  occur in arm A, so removing them would not reduce the relative overhead.
- Approximate attention for the proxy entries.

## 5. Interpretation

- These estimates are not measurements. Expected order of benefit:
  1. **P7:** 1.1, then 1.3 and 1.2.
  2. **P6-iso and P7-simple:** 2.4, 2.2 and 2.1, then 2.3 and 2.5.

  1.1 also applies to P7-simple.
- Shared-code changes (2.x) can also change A's timing only where A executes the
  same path. A has no targets, so A's time should be essentially unchanged.
  Re-time A on the same code version anyway before computing time-matched steps.
- "Exact" items preserve bitwise reproducibility of existing runs and can be
  checked against completed checkpoints. "Equivalent" items need fresh runs
  for any comparison.

## 6. Implementation decision and local verification

Implemented the limited execution-only items **1.2, 2.1 and 2.3**:

- Convolution masks are converted once per forward/dtype and reused across
  layers and checkpoint replay. Each lag adds into the owned accumulator slice;
  the lag order, source detachment, convolution parameters and EMS term remain.
- Clipping counts use the Trainer's existing logging-step decision, including
  `logging_first_step`. Non-step logging (including epoch logging) conservatively
  retains collection on every step. Direct model calls preserve their previous default.
  Every microbatch still contributes all normalization moments, and the same
  four tensors follow the same distributed reduction path. On unlogged steps
  the clipping-count tensor is zero, but no false zero `clip_fraction` is logged.
  If neither auxiliary loss nor clipping counts are needed, normalization is
  skipped while moments are still collected. No auxiliary loss is subsampled.
- MLP target windows keep the first contribution as before, allocate private
  storage on the second contribution, and add subsequent contributions in place.
  Thus detached decoder outputs and overlapping windows cannot be mutated by
  accumulation. Addition order and weighted/layer-normalized targets remain.
  P3's separate incremental-target aggregation is unchanged.

**Deferred:** 1.1, 1.3, 2.2, 2.4 and 2.5. These change kernel execution and/or
floating-point reduction order, and require their proposed CUDA numerical and
performance gates. No LSE attention merge, compiled function, custom backward,
precision change, new training option or checkpoint field was introduced.

Local evidence under `temp/proxy-safe-speed-20261008/` (Git-ignored):

- `reference-check.log` / `reference-check.json`: **48 exact full-depth cases**
  against a saved copy of the pre-change implementation. Twelve P6/P7 arms,
  28 tiny decoder blocks, FP32/BF16, SDPA/FA4 CPU oracle and checkpointing.
  Every output (including statistics), parameter gradient and state tensor
  matches with `rtol=atol=0`.
- `optimization-tests.log`: five expanded reference/replay/resume checks passed
  in 32.199 s, now including original P7, P7-kq, P7-ems and P7-mlp convolution.
- `speed-tests-final-logging.log`: three new tests passed in 6.265 s. They check short
  sequences and document masks, clipping disabled versus enabled, and actual
  Trainer runs with accumulation, first-step logging on/off and epoch logging. Training losses,
  model/normalization state and AdamW state match the always-count-clipping
  reference exactly. Initial test-only mistakes (identity of an empty tuple,
  and a 64-token callback budget for a 16-token batch) were fixed before this
  passing run; production checks were not relaxed. The final convolution test
  includes native BF16 parameters as well as FP32 parameters with BF16 autocast;
  the earlier 4.612 s and 4.872 s passes preceded the native-BF16 and epoch-logging
  coverage, respectively.
- `regression.log`: **91 tests passed in 675.339 s**, covering both proxy
  families, earlier P1/P3 heads, target normalization, checkpoint replay,
  two-rank CPU DDP, gradient accumulation, exact Trainer resume, evaluation and
  supervised fine-tuning. Together with the five optimization and three final
  speed-safety checks, **99 tests passed** across the selected invocations.
  The final non-step logging guard is exercised by `speed-tests-final-logging.log`.

Final focused command (same CPU environment as the regression):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  temp/p6-review-env/bin/python -m unittest tests.test_proxy_speed
```

`operator-counts.json` profiles one CPU forward/backward without checkpointing,
including target statistics on a non-logging step (28 tiny blocks, SDPA/FP32).
P6-iso out-of-place `aten::add` calls fall from 372 to 348, replaced by 24
in-place adds; clipping comparisons fall from 12 to zero. Original P7 padding
calls fall from 73 to one and `_to_copy` calls from 888 to 795. Its new backward
also introduces 36 `CopySlices` calls. This confirms removed work and a changed
backward allocation pattern; it does **not** establish net GPU throughput gain.

The eight-GPU baseline profile proposed in section 0 has **not** been run.
These are correctness results, not evidence of a B200 speedup. No B200 access,
deployment, environment changes or new experiments occurred. Actual CUDA
kernel correctness and full-size throughput still need verification before
deploying these changes to a GPU research queue.
