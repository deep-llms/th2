# B-DOT dispatch overhead: investigation and fixes

Status: **GPU verification and before/after timing passed** on B200 6Y3O.
Completed 2026-10-20; no research pilot launched.
Design: [B-DOT v1.4](BDOT_v2_spec.md). Architecture, training objectives, FiLM,
router balancing, exploration, expert parameters, optimizer and data are unchanged.

## Measured result

Two repeats per case, reversing old/new order on the second repeat. Each repeat
uses five warmup and 50 measured updates through HF Trainer, all eight B200s,
BF16/FA4, 12 layers, sequence 1,024, microbatch 16 × accumulation 4. Each update
contains 524,288 input tokens. Timing includes forward/backward, accumulation,
gradient clipping, AdamW, scheduler and routing bookkeeping.

| Path | Before, seconds/update | After | Reduction in update time |
|---|---:|---:|---:|
| Native Qwen reference | 0.6176 | — | — |
| Fixed-route D, reference dispatcher | 0.7070 | 0.6646 | 6.0% |
| Hard routing, grouped dispatcher | 0.7722 | 0.7332 | 5.1% |
| Soft routing, grouped dispatcher | 0.9992 | 0.9568 | 4.2% |

These correspond to throughput gains of 6.4%, 5.3% and 4.4%, respectively.
Observed old/new ranges do not overlap in any of these three pairs.
New hard routing remains **18.7% slower than native Qwen**, or **10.3% slower
than fixed-route D**. New soft routing remains **54.9% slower than native**.
This is a useful, modest improvement; it does not remove routing's real costs.

The new reference dispatcher takes 0.7780 seconds for hard and 0.9589 for soft.
Grouped still helps hard routing; its soft advantage is too small here to treat
as a clear result. The configured default remains reference; backend selection
must stay fixed within a training/resume comparison.

The data are synthetic with 128-token document fragments. Preprocessing,
validation and checkpoint saving are excluded. These are full-update timings,
not full-run training time, convergence evidence or a statistical confidence
interval from two repeats. The soft phase performs two expert/attention branches;
it cannot be made equivalent to one branch merely by removing dispatch overhead.

## What the original implementation was doing

An eight-B200 profile used the existing FA4/train environment, 12-layer Qwen
dimensions, sequence 1,024, microbatch 16 and accumulation 4. Rank zero recorded
one update after five warmup updates, through the actual HF Trainer.

| Path | Synchronizations, before → after | Kernel launches, before → after |
|---|---:|---:|
| Native Qwen | 11 (reference) | 8,652 (reference) |
| Fixed-route D | 224 → 28 | 12,217 → 11,287 |
| Hard routing, grouped | 260 → 96 | 18,928 → 18,254 |

These are profiler event counts, not percentages of wall time. Profiler
instrumentation changes timing; nested CPU/CUDA durations must not be added as
independent costs. Unprofiled repeated full-update benchmarks determine speed.

Code inspection identifies unnecessary work corresponding to these events:

- Fixed/home-only steps used the generic sort/gather/scatter path despite the
  expert being known for every token.
- Hard straight-through gates were filtered with CUDA `nonzero`, even though
  their finite forward values are exactly one.
- CUDA `bincount` determines the input range before allocating its result,
  even with `minlength`. Our valid expert IDs already have a known range.
- The reference dispatcher repeatedly copied full output buffers for each expert.
- The two-input RoPE helper was called with the same tensor twice, then one
  output was discarded, separately for K and for each Q branch.
- Last-update diagnostics were transferred to the CPU/formatted every update,
  even when no log was requested.

## Changes

`bdot/model.py` skips route discovery when every token must use its home expert.
It retains the original expert gathers/scatters and gate expression, including
nonzero learned FiLM. Hard straight-through dispatch skips
zero-gate filtering; soft and probability-scaled gates retain filtering, including
zero/underflow cases. Counts use fixed-size integer scatter-add. Top-1 uses
first-maximum `argmax`, preserving the lowest-ID tie break; top-2 keeps stable
sorting. Reference projections are concatenated and scattered once per stage.
Each Q/K tensor is rotated once using the same native formula.

Balancing, global loss accounting and step counters still update every optimizer
step. Diagnostics retain a small detached snapshot and format it only when read.
Logging returns the same keys and values. Parameter names and checkpoint tensor
state are unchanged. Existing strict source/runtime resume guards remain active;
this is not permission to bypass them for older runs.

Grouped GEMM remains optional. Both new reference and new grouped execution are
benchmarked, since improving dispatch can change their relative performance.

## Verification and evidence

- Final complete CPU B-DOT suite: **48 tests passed** in 321.35 seconds.
- Final focused tests of the conservative home path: 20 passed, including new
  nonzero-FiLM, straight-through gate-gradient and lazy-diagnostic tests.
- Eight frozen-old/new CPU comparisons of the current revision, three AdamW
  updates each: routes and final weights match exactly.
- All **32 CUDA comparisons** passed (SDPA/FA4 × reference/grouped × eight
  routing configurations), three AdamW updates each: **zero gradient and final
  parameter differences** against the frozen old package in these fixtures.
- Checkpointed two-rank DDP checks passed for home, soft and no-reuse routing.
  Their comparison to single-process accumulation has the expected small BF16
  reduction differences (maximum absolute gradient difference 0.001465).
- Both eight-rank curriculum A/B resume suites passed: exact restored/final
  model, optimizer and scheduler state; identical data/RNG/routes and zero loss
  difference on every rank. These are reduced-width, 12-layer fixtures.
- All 18 unprofiled full-width timing jobs and three updated profiles passed.
  These checks do not replace the separate full-width pilot acceptance tests.

The first three attempts failed a strict multi-update GPU comparison and stopped
before timing. Initial forwards matched; a small post-AdamW weight difference
preceded a larger BF16 forward difference on the second update of a grouped,
fixed-route case. Preserving flattened tensor shapes did not resolve it. The
accepted revision removes the direct expert-computation shortcut and retains its
original autograd path. The revised GPU comparison passes without relaxing any
tolerance. The rejected shortcut is not included. All three failed queues
restored verified burns.

The old package was frozen and hashed before changes at
`/mnt/local/_outputs/deep-llms_th2/6y3o-dot-dispatch-profile-20261020-a01/reference`.
Profile launch: `5062ed2`. Current verification launch: **`a5f65aa`**, root
`/mnt/local/_outputs/deep-llms_th2/6y3o-dot-dispatch-opt-20261020-a04`.
Earlier attempts have separate a01–a03 roots and are retained for diagnosis.
Local raw evidence: `temp/dispatch_opt_20261020/`.
All 14 requested result files were downloaded to its `results/` subfolder.
The complete summary's SHA256 matches the remote status receipt and completion
manifest. See the [durable receipt](results/bdot_dispatch_optimization_20261020.json).

The optimization queue restored communicating burns at **16:47:00 UTC**.
An independent observation at **18:25 UTC** verified all eight worker identities
(125740–125747), released ownership guard and advancing collective cycles
6,840 → 6,850. Accelerate was copied to its cache and verified with `accelerate env`
before launch; only known burn workers were reclaimed, and all eight GPUs were
verified free before testing. No environment, driver, original dataset or
existing research checkpoint was modified.
