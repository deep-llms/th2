# SDPA attention backend: what is verified and what to check

Date: 2026-10-04. Status: **C1/C3 implemented and B200 profiling completed; see Section 8.**
Sections 1–6 preserve the original checklist and pre-implementation review.

> **Read Section 6 before implementing any check.** It corrects several
> instructions in Sections 1–5:
> - the backward used for C1 profiling;
> - how C1 identifies each call site;
> - the FP32 reference batch size in C2;
> - the fields recorded in C3;
> - the meaning of pinning in C4;
> - the cost unit of C5;
> - the bottleneck arm count.
>
> Where they conflict, Section 6 takes precedence.

This document lists the checks still needed to know which attention kernels the
Deep-KV training code actually uses on B200, and to keep that choice safe over
time. It is self-contained.

## 1. Background

Training uses PyTorch `scaled_dot_product_attention` (SDPA) through two call
sites:

| Call site | Code | Mask passed to SDPA |
|---|---|---|
| Backbone self-attention (every block, including the custom block-5 and block-21 paths in `special_block`) | Hugging Face `sdpa_attention_forward`, via `ALL_ATTENTION_FUNCTIONS` | Additive float mask from `Context.additive_mask(hidden.dtype)`, shape `[batch, 1, 2048, 2048]`. It is built in FP32 and cast to BF16 under autocast. Hugging Face repeats K/V to 16 heads (`repeat_kv`) whenever a mask is given. |
| Auxiliary strict-past branch (arms B–G and both bottleneck arms; the bottleneck consumer-aware loss also calls it) | `AuxiliaryKV.forward` in `deep_kv/model.py` | **Boolean** mask (`strict \| empty-row dummy`), shape `[batch, 1, 2048, 2048]`. K/V repeated to 16 heads with `repeat_kv`. |

The routing losses of arms F/G do not call SDPA; they use explicit FP32 matmuls.

SDPA is not one kernel. PyTorch dispatches each call to one of several backends:

| Backend | Profiler operator name |
|---|---|
| FlashAttention | `aten::_scaled_dot_product_flash_attention` |
| Memory-efficient | `aten::_scaled_dot_product_efficient_attention` |
| cuDNN | `aten::_scaled_dot_product_cudnn_attention` |
| Math (unfused reference) | `aten::_scaled_dot_product_attention_math` |

The choice depends on GPU, PyTorch/cuDNN versions, dtype, shapes, mask presence
and mask dtype. It can change silently when any of these change.

## 2. What is already verified

1. **Attention-only profile on B200.** This ran on node tjx3 with
   torch 2.14.1+cu130, BF16, shape `[2, 16, 2048, 128]`, a BF16 additive mask,
   and Q/K/V all with 16 heads (no GQA inside the kernel). Implicit causal,
   explicit causal and document-isolated masks **all dispatched to cuDNN**:

   | Mode | Forward + backward |
   |---|---:|
   | Implicit causal | 0.2535 ms |
   | Explicit causal | 0.6238 ms |
   | Document-isolated | 0.6239 ms |

   Evidence: `temp/tjx3-document-a03.log`, around line 58.
2. **Correctness of the dense backbone path on B200**, whatever kernel it used:
   - Document-isolation tests passed for all 11 arms in FP32 and BF16.
   - Same-weight comparisons against an FP32 math-SDPA reference (TF32
     disabled) differed by 0.54–1.21% in gradient relative L2. This is ordinary
     BF16 error.
3. **Correctness of the auxiliary branch.** CPU tests compare it with a
   hand-written reference attention, including strict-past masking, empty rows,
   causality and gradients. These test semantics, not which GPU kernel runs.

## 3. What is not verified

| Gap | Why it matters |
|---|---|
| **G1.** Backend actually used inside the full model during training | The profile in §2.1 used batch 2 and a BF16 mask. Production uses microbatch 16, a mask built in FP32 and cast under autocast, and the Hugging Face call path. Dispatch is likely the same, but this was never observed in a training process. |
| **G2.** Backend of the auxiliary branch | It passes a **boolean** mask, which was never profiled. It may use cuDNN, memory-efficient or math. |
| **G3.** Backend used by the historical A–G runs | They ran on node 78gg with torch 2.14.0+cu130 and cuDNN 9.24.0.43. No per-call profile exists from that node, and the node has been replaced, so this gap can only be documented, not closed. |
| **G4.** Stability across version changes | A PyTorch, cuDNN or driver upgrade can change dispatch or kernel behavior without any code change. |

## 4. Checks to add

### C1. Profile every SDPA call in the full model (closes G1 and G2)

Run inside the next production smoke, on the real B200 runtime, for every arm
in the queue. Use one training microbatch at the production shape (microbatch
16, sequence 2,048, BF16 autocast), forward and backward, with the same
`Context` (and document segments, if isolation is enabled) as training.

```python
import torch
from torch.profiler import ProfilerActivity, profile

SDPA_OPS = ("_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
            "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_attention_math")

def sdpa_backends(model, context):
    model.train()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as trace:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(context)
        out["lm_sum"].backward()
    torch.cuda.synchronize()
    counts = {}
    for event in trace.key_averages(group_by_input_shape=True):
        if any(op in event.key for op in SDPA_OPS):
            key = (event.key, str(event.input_shapes))
            counts[key] = counts.get(key, 0) + event.count
    return counts
```

To separate backbone from auxiliary calls, also call `model.aux(...)` alone
with production-shape inputs (normalized states, rotated query, rotary tables
and `context.allowed()`) under the same profiler, and record its operators.

**Pass criteria:**
- No call falls back to `_scaled_dot_product_attention_math` at production
  shape. A math fallback is correct but slow, and should be a known decision.
- Per forward pass there are 28 backbone SDPA calls. B–G and the bottleneck
  arms add one auxiliary call, and the consumer-aware loss adds one more. With
  activation checkpointing enabled (`checkpoint_layers`, `checkpoint_aux`),
  checkpointed layers rerun their forward during backward, so total counts are
  higher. Compare counts against the expected value for the recipe's
  checkpoint settings, rather than a fixed number.
- The backend for each call site is recorded in the run receipt (see C3).

### C2. Same-weight reference check after any runtime change (covers G4)

Run whenever PyTorch, cuDNN, CUDA, the driver or Transformers changes, before
production training:
1. The document-isolation tests (`tests/test_document_isolation.py`) on B200 in
   FP32 and BF16, all arms.
2. A same-weight comparison of loss, final hidden states and full parameter
   gradients for one production-shape microbatch: BF16 production backends
   versus FP32 math SDPA with TF32 disabled. Use the existing limits: gradient
   relative L2 < 3%, hidden/logit relative L2 < 2%, loss difference < 0.01.

### C3. Record the runtime in every run receipt (covers G3 going forward)

Add to each run's preflight or receipt:
- `torch.__version__`, `torch.version.cuda`, `torch.backends.cudnn.version()`,
  and the driver version.
- The backend per call site from C1.
- The SDPA backend flags in effect:
  `torch.backends.cuda.flash_sdp_enabled()`, `mem_efficient_sdp_enabled()`,
  `cudnn_sdp_enabled()`, `math_sdp_enabled()`.

A later comparison between runs can then show whether they used the same
kernels.

### C4. Optional: pin the backend explicitly

For strict reproducibility across environments, wrap training forward/backward
in `torch.nn.attention.sdpa_kernel([...])` with an explicit backend list,
for example `[SDPBackend.CUDNN_ATTENTION]`. Caveats:
- If the pinned backend cannot handle a call (for example, a mask type it does
  not support), PyTorch raises an error instead of falling back. Run C1 first to
  confirm the chosen backend supports both call sites.
- Pinning changes nothing numerically if dispatch already selects that backend.
  It only prevents silent changes later.

### C5. Same-backend trajectory control (related to the FA4 comparison)

To calibrate how much two runs on the **same** backend diverge, rerun dense SDPA
isolation for 200 updates with the same seed, data and recipe as the existing
dense run in `artifacts/document-stability-200-20261004-a01/`. Compare:
- update-by-update loss and gradient norm;
- the first update where gradient norms differ by more than 0.1%;
- the late-run gap distribution.

This shows whether the dense-versus-FA4 gaps (up to 48% in gradient norm) are
within ordinary same-backend variation. See Section 9 of
`DOCUMENT_ISOLATION_INVESTIGATION_20261004.md` for the existing per-update
analysis.

## 5. Priority

| Check | When | Cost |
|---|---|---|
| C1 | Next production smoke | Seconds per arm |
| C3 | Next launch onward | Negligible |
| C2 | After any runtime/version change | Minutes |
| C5 | Before adopting FA4 | One 200-update run (~10–15 GPU minutes) |
| C4 | Optional, after C1 | Negligible |

## 6. Review addendum: corrections before implementation

Added 2026-10-04. This review preserves the original checklist above. Where it
qualifies an earlier instruction, use the clarification below when implementing
the checks. No new GPU tests or training were launched for this review.

The overall priority is sound: **implement C1 and C3 first**. The existing
microbenchmark confirms cuDNN for its tested inputs, but does not establish
which backend every backbone and auxiliary call uses in real training. Profile
those calls before choosing whether to restrict backend selection.

### 6.1 C1: profile the actual objective and identify each call site

The example's `out["lm_sum"].backward()` is incomplete for custom arms. It
backpropagates the LM term only. In particular, the Consumer-Aware extractor
objective has a detached consumer path whose backward computation is not
reached by the LM loss. Other auxiliary losses and their checkpoint
recomputation are also omitted from this backward profile.

Use the actual objective construction in
[DeepKVTrainer.compute_loss](../deep_kv/training.py), including the arm's loss
weights and normalization. Clear gradients before the diagnostic. Profile a
representative training microbatch without an optimizer update, using the real
precision and checkpoint settings. If profiling inside a smoke run, ensure the
diagnostic does not leave gradients or alter the subsequent training RNG stream.

Record backbone, real auxiliary branch, and detached consumer calls separately
inside the real model execution, using named profiler scopes or equivalent
instrumentation. Operator/shape aggregation alone can merge calls with the same
shape. A standalone auxiliary test is useful supplementary evidence, but does
not replace observing its actual inputs and execution context in the full model.

Count forward and backward operators separately. Derive expected counts from
the model configuration and active arm; account for checkpoint recomputation.
The current 28-layer configuration does not justify hard-coding that count for
future models. Treat a math fallback as a performance finding requiring review,
not automatically as a numerical correctness failure. Warm up before profiling;
the profiler itself is not a throughput benchmark.

Profile the **evaluation path** too, not only training. The reported held-out
LM loss comes from evaluation, which runs under `torch.no_grad()` with
`model.eval()` and `per_device_eval_batch_size` contexts per rank. SDPA can
dispatch differently when no gradient is required. Record one evaluation
microbatch under the same named call-site scopes, at the real evaluation batch
size and precision, separately from the training profile.

### 6.2 C2: use a capacity-verified FP32 reference

Full-model FP32 math SDPA at microbatch 16 and sequence length 2,048 has not been
capacity-verified. The successful trained-checkpoint FP32 reference checks used
microbatch **2**, while BF16 backend comparisons also covered microbatch 16.
Do not assume that a BF16 batch fitting in memory means the corresponding FP32
math-attention batch will fit.

Keep production-shape BF16 profiling in C1. Start FP32 comparisons at the verified
smaller batch, with identical weights, inputs, masks and objective for the two
precision paths. Record that this validates the smaller shape, not production
microbatch-16 FP32 behavior. Increase the reference batch only after a separate
capacity check.

For custom arms, compare the full objective and relevant auxiliary losses and
gradients, not just LM loss. Preserve the stated tolerances when evaluating a
check; report failures rather than silently relaxing them. The existing baseline
results do not establish that every custom arm has already passed these limits.

### 6.3 C3: record configuration as well as backend names

Add the following alongside the runtime versions and enabled-backend flags:

- Transformers version and the model/arm configuration.
- Q/K/V shapes and dtypes, mask shape and dtype, and `is_causal` for each call site.
- Autocast precision, TF32/determinism settings, and activation-checkpoint settings.
- Observed forward/backward backend operators, tagged by call site.

Distinguish the mask as constructed, the arguments reaching the SDPA API, and
any conversion under autocast. Record observed dtypes instead of inferring them
from the construction code alone. Enabled-backend flags describe available
choices; only the profile establishes which choice a particular call used.

These receipts improve comparisons between environments. They do not establish
that a backend family used an identical internal algorithm across versions.

### 6.4 C4: pinning controls dispatch, not strict reproducibility

Replace the rationale “for strict reproducibility across environments” with
“to prevent unintended backend-family changes.” An explicit
`sdpa_kernel([SDPBackend.CUDNN_ATTENTION])` restriction can prevent fallback to
another backend family, but does not freeze cuDNN's internal implementation,
algorithm selection, or backward repeatability across runtime changes.

Even with the same backend selected, do not promise bitwise equality. Our
same-weight tests observed small backward differences when repeating the full
pipeline without changing its backend. The statement that pinning “changes
nothing numerically” should therefore mean that it introduces no intended
backend-family change when dispatch already selected that family; it is not an
exact numerical guarantee.

PyTorch documents [SDPA backend selection](https://docs.pytorch.org/docs/2.14/generated/torch.nn.attention.sdpa_kernel.html)
separately from [reproducibility limitations](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
Verify every call site's support before restricting the backend, and keep
runtime/reference checks after any version change.

### 6.5 C5: correct the cost unit and limit the inference

The estimated 10–15 minutes is **elapsed time on eight GPUs**, equivalent to
approximately **80–120 GPU-minutes**, not 10–15 GPU-minutes. Startup, evaluation
and recovery overhead can affect elapsed time.

One additional dense run is a useful same-backend control. It can show whether
that particular pair develops comparable gaps, but cannot establish a reliable
distribution of ordinary trajectory variation. Multiple repeats would provide
stronger evidence. Do not assume in advance that divergence will begin at about
update 100 or reach 48%.

Preserve the original data order, seed, optimizer, LR schedule and runtime when
forming the control. Compare held-out behavior as well as per-update losses and
gradient norms. If a fresh profile leads to changing the dispatch policy, that
is a changed experiment configuration and must be recorded explicitly.

### 6.6 Arm coverage and recommendation

“Both bottleneck arms” means two **families**, each with two variants. The code
currently defines four bottleneck variants: `Task-Aware-NoAlign`,
`Task-Aware-Align`, `Consumer-Aware-NoAlign`, and `Consumer-Aware-Align`. Together
with A–G, this makes 11 arms. State the actual variants covered by each check.

Proceed with corrected C1 and C3 during the next authorized production smoke.
Keep C2 memory-safe and apply it after runtime changes. C4 remains optional;
C5 is an additional trajectory control, not a prerequisite for answering which
SDPA kernels the current model invokes. This checklist and review do not
themselves authorize a new GPU run or change the production attention policy.

## 7. Implementation started (2026-10-04)

The user authorized execution after rereading Section 6, including its added
requirement to profile evaluation. C1/C3 are implemented first; B200 verification
is pending at this point in the record.

- `deep_kv/sdpa_audit.py` temporarily labels real SDPA calls and observes profiler
  operators. Forward scopes and autograd sequence/thread IDs associate fused
  backward kernels with their originating call sites. Checkpoint recomputation
  is reported separately. API tensor metadata and training saved-tensor metadata
  distinguish arguments from tensors retained after autocast. Evaluation saves
  no tensors, so it does not claim a saved-tensor observation there.
- `DeepKVTrainer` records the first actual CUDA training and evaluation
  microbatches on rank zero of every invocation. It retains the original loss,
  Trainer loop and backend selection. `train.py` links the resulting receipt
  filenames in `result.json`. These first-microbatch receipts include startup;
  they are dispatch evidence, not timing benchmarks.
- `scripts/profile_sdpa_backends.py` warms up and profiles the full Trainer
  objective for all 11 arms at microbatch 16 / sequence 2,048, training and
  evaluation, cross-document and isolated masks, checkpointing on/off. Each GPU
  handles independent cases; the full-model checks make no optimizer updates.
  A separate tiny Consumer-Aware Trainer smoke makes two updates to verify the
  automatic receipt integration. Production document isolation remains disabled.
- `tests/test_sdpa_audit.py` checks unchanged loss/gradients/weights/RNG, complete
  call attribution including detached consumer backward, both checkpoint modes,
  evaluation, and wrapper restoration after an exception.

C4 remains optional and is not enabled. C5 is deferred while answering the
backend-dispatch question. Full-model FP32 micro16 is not attempted; the earlier
same-weight FP32 evidence remains explicitly scoped to micro2 baseline checks.

## 8. Verified C1/C3 results (2026-10-04)

**All 88 full-model profiles passed. Every observed SDPA forward and fused
backward used cuDNN; no math, FlashAttention or memory-efficient dispatch was
observed.** This includes evaluation, the boolean-mask auxiliary branch, and
the detached consumer path. It answers G1/G2 for the tested current runtime;
it cannot recover the historical node's dispatch (G3).

### Coverage and execution

Implementation/launch commit: `ab62618`. Job: `sdpa-full-profile-20261004-a01`.
Completed at 15:36:08 UTC on `thiennh-p6-tjx3-worker-0`.

The 88 cases are 11 arms × 2 checkpoint settings × 2 mask modes × 2 phases:
A–G and all four Task-/Consumer-Aware variants; all activation checkpoints on
or off; explicit cross-document or dense document-isolated masks; training
forward/backward or evaluation without gradients. Both phases used microbatch
16, sequence length 2,048, and CUDA BF16 autocast. Eight GPUs processed
independent cases, using the full production model and Trainer objective.
This was not an eight-rank DDP throughput measurement.

Full models used random initialization (seed 42), including the production
zero-initialized auxiliary output. Inputs came from the existing real English
benchmark pool. No full-model optimizer updates occurred; all before/after
parameter fingerprints matched and all observed gradients were finite. This
is dispatch evidence, not a same-weight numerical comparison or convergence
validation of every arm. CPU preservation tests separately activate the
auxiliary output and verify that auditing leaves loss/gradients/weights/RNG
unchanged.

Runtime: torch `2.14.1+cu130`, CUDA `13.0`, cuDNN package `9.24.0.43`
(runtime integer `92400`), Transformers `5.9.0`, Accelerate `1.13.0`, driver
`580.167.08`. All four SDPA backend flags were enabled. Matmul TF32 was off;
cuDNN TF32 was on. Deterministic algorithms, cuDNN deterministic mode and
cuDNN benchmarking were off. These flags describe this test, not a guarantee
of repeatability.

### Observed calls and dtypes

| Site | Q / K / V at SDPA API | Mask at API | Train forward/backward | Eval forward |
|---|---|---|---|---|
| Every backbone layer | FP32 / FP32 / BF16 | FP32 additive | cuDNN / cuDNN | cuDNN |
| Auxiliary branch | BF16 / BF16 / BF16 | Boolean | cuDNN / cuDNN | cuDNN |
| Detached consumer (consumer-aware variants) | BF16 / BF16 / BF16 | Boolean | cuDNN / cuDNN | cuDNN |

All API Q/K/V shapes were `[16,16,2048,128]`; masks were
`[16,1,2048,2048]`. Every call had `is_causal=False`, `enable_gqa=False`,
and BF16 autocast enabled. Explicit masks express causality/isolation.

**This corrects the inference in Sections 1 and 3 about mask dtype.** The
backbone mask reaches the API as FP32 in this execution. With checkpointing off,
the training saved-tensor observations include BF16 tensors of the attention
input shape and BF16 tensors of the mask shape. The auxiliary boolean API mask
likewise corresponds to a retained BF16 tensor of that mask shape. Saved-tensor
metadata has no semantic labels; it records what autograd retained inside SDPA,
not every internal conversion. Evaluation saves no tensors, so it has only API
dtype and operator observations. Do not infer FP32 attention arithmetic merely
from FP32 arguments before autocast.

Counts matched exactly in every case:

| Arm family | Initial forward calls | Recomputed calls with checkpoints on (train) | Fused backward calls (train) |
|---|---:|---:|---:|
| A | 28 | 28 | 28 |
| B–G, Task-Aware variants | 29 | 29 | 29 |
| Consumer-Aware variants | 30 | 30 | 30 |

Checkpointing off produced no recomputation. Evaluation produced neither
recomputation nor backward calls in either checkpoint setting. All calls and
fused backward operators were attributed to their sites. Across the 88 cases:
3,200 `aten::_scaled_dot_product_cudnn_attention` and 1,280
`aten::_scaled_dot_product_cudnn_attention_backward` events.

### Automatic receipts and validation

`DeepKVTrainer` now records one first-training and one first-evaluation CUDA
microbatch on rank zero per invocation, including a resumed invocation. It
observes the real calls, preserves backend selection, and adds no extra
forward or optimizer update. Receipts live as `sdpa-train-*.json` and
`sdpa-eval-*.json` in the run directory; `train.py` lists them in `result.json`.
They include runtime/configuration, API tensor metadata, checkpoint status,
call attribution and observed operators. Profiling adds one-time overhead;
these receipts are not throughput measurements.

A separate tiny Consumer-Aware-Align model completed two actual HF Trainer
updates on B200, including evaluation before training and afterward. It wrote
exactly two automatic receipts, with complete attribution. This verifies the
single-GPU Trainer integration; the new receipt hook has not yet been exercised
in a full eight-rank training launch. Full-model cases exercise all arms through
`DeepKVTrainer.compute_loss`, independently on each GPU.

Local validation passed: two audit tests covering all 11 arms and both
checkpoint settings, nine document-training/trained-attention regression tests,
and ten Trainer tests. B200 preflight passed the audit plus nine regression
tests (11 tests). Shell syntax and Python compilation checks also passed.

### Evidence and GPU handoff

Local evidence: `artifacts/sdpa-full-profile-20261004-a01/`, especially
`result.json`, `profiles/summary.json`, the 88 case JSON files, and
`profiles/trainer-receipt-smoke.json`. All 104 exported evidence files passed
their size/SHA-256 manifest checks. Archive SHA-256:
`ebc23667459a57a6739bb03a99b68bc8d62c6ee44238158974502ae3f679cba1`.

Remote root: `/mnt/local/_outputs/deep-llms_th2/sdpa-full-profile-20261004-a01`.
The launch copied and byte-verified the resources Accelerate config, printed
`accelerate env`, stopped only the eight verified burn workers, and verified
all eight GPUs free. Burns restarted automatically after profiling; collective
progress was verified at 15:36:08 UTC. The read-only follow-up at 15:39:13 UTC
confirmed workers 21008–21015 on GPUs 0–7 at 100% utilization, with the guard
released. Evidence logs: `temp/sdpa-full-profile-launch-a01.log` and
`temp/sdpa-full-profile-monitor-a01.log`.

### Remaining decisions

C1/C3 are complete for this matrix and integrated for future runs. Production
document isolation remains disabled, and backend selection is not pinned.
C2 remains a runtime-change gate; earlier baseline FP32 reference evidence is
microbatch 2, not all-arm or microbatch-16 validation. C4 is optional. C5, the
additional 200-update same-backend trajectory control, was not run here and
remains useful before reconsidering FA4. These dispatch results do not resolve
the previous dense-versus-FA4 trajectory gap or promise bitwise reproducibility.

## 9. Production choice (2026-10-05)

The user subsequently selected dense SDPA document isolation. `train.py` and the
shared recipe now enable it by default for both phases and every arm, using the
same boundary semantics as these tests. See [the training guide](DEEP_KV_TRAINING.md#eos-packing-and-document-isolation).
Section 8's statements about production isolation being disabled describe the
state at the time of that audit. This code change does not launch training or
pin the SDPA backend. The FA4 trajectory-control experiment remains unrun.

Validation: 31 local tests passed (real entry-point train/eval for all 11 arms,
cache reuse/rebuild and token preservation, isolation semantics, bottleneck losses,
resume, and audit regressions). Eight-process CPU BF16 micro16/accum4 checks for
A and Consumer-Aware-Align passed: resumed versus uninterrupted max parameter
difference 9.313225746154785e-10; masked Consumer-Aware DDP versus global-batch
reference max difference 3.725290298461914e-09. Python compilation/diff checks
passed. Logs: temp/production-isolation-tests-a02.log and
temp/production-isolation-ddp-a01.log; receipt:
temp/deep-kv-isolated-resume-a01/resume_verified.json. These local integration
checks do not replace a future B200 launch preflight.
