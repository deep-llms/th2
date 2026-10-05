# Proxy heads (P1/P3): implementation review findings

Date: 2026-10-05. Reviewed against `docs/proxy_heads_P1_P3_spec_v3.md`
(Revision 4), at commit `86fd237`. Code reviewed: `deep_kv/proxy.py`,
`deep_kv/proxy_training.py`, the proxy and document-isolation paths in
`train.py` and `deep_kv/packing.py`, `deep_kv/__main__.py`, `deep_kv/report.py`,
and `scripts/calibrate_proxy_mask.py`. No GPU work was run for this review.

## 1. Verdict

The model and training implementation matches the specification:
- §2.1 architecture checks;
- §2.3 document isolation;
- §3 proxy KV groups and gated injection;
- §4 P1;
- §5 P3, including an exact chunkwise scan;
- §6 normalization and centering;
- §7 λ schedule and the λ = 0 control;
- §8 compute matching, with widening +72 for V1 and +88 for V3;
- §10 logging and the reliance ablation.

All 18 tests in `tests/test_proxy_heads.py` and `tests/test_proxy_training.py`
pass on CPU, including T1–T8 with the specified tolerances and all six T4
boundary layouts.

The completed isolated baseline A (seed 42, 2,500 updates, held-out LM loss
3.4779) ran through `ProxyModel`/`ProxyTrainer` (`proxy_screen=True`), the
same attention path as the proxy arms.

The issues below concern running the screen, not the model's computation.
F1 and F2 must be resolved before P1/P3 runs. F3 should be resolved before
launch.

## 2. Findings

### F1. Channel-mask calibration cannot load the only available trained checkpoint

**Status:** blocking for computing the mask `M` (§6).

§6 asks for `M` to be computed once on a trained vanilla checkpoint.
`scripts/calibrate_proxy_mask.py` accepts only an original `DeepKV` arm A:
- It builds `DeepKV.from_scratch(cfg, 'A', ...)`.
- `calibrate()` raises if `model.proxy_screen` is set or the arm is not `'A'`.
- It loads weights with `scripts/check_trained_attention.restore`, which calls
  `load_state_dict(..., strict=True)`.

The original non-isolated arm-A checkpoints were on the replaced 78gg node and
are not available. The only trained vanilla checkpoint now is the isolated
baseline A. It was trained as a `ProxyModel`, whose state dict has four extra
buffers. Reproduced locally with a small model:

```text
keys in ProxyModel-A state but not DeepKV-A: ['channel_mask', 'gamma', 'mu', 'mu_initialized']
strict load fails: Error(s) in loading state_dict for DeepKV:
Unexpected key(s) in state_dict: "mu", "mu_initialized", "gamma", "channel_mask".
```

The failure is loud, not silent. But as the code stands, `M` cannot be computed
from any checkpoint that exists.

**Options:**
1. **Recommended.** Let the script load a `ProxyModel` arm-A checkpoint. Either
   build `ProxyModel` arm A directly and relax the `proxy_screen` check for
   arm A, or drop exactly those four buffers after verifying they hold arm A's
   default values. Add a regression test with a saved `ProxyModel` arm-A
   checkpoint.
2. Use the empty mask and log it. §6 allows this only when *no* checkpoint is
   available, which is no longer the case.

Whichever is chosen, every proxy arm must use the same mask file, including
the λ = 0 controls.

### F2. Matching the proxy arms to the existing baseline A

**Status:** must be decided before generating the P1/P3 queue.

`report()` treats arms as matched only if their full `train_config.json` is
equal, apart from the arm name (`deep_kv/report.py`, around lines 19–35). That
config includes:
- every `pilot` field, including `checkpoint_layers`, `checkpoint_lm` and
  `checkpoint_aux` (`train.py`, around lines 214–219);
- `proxy_mask`, the mask receipt (`train.py`, around lines 176–189 and 223–224).

The baseline A launch (`3eec7f9`) differs from the defaults in both:

| Setting | Baseline A | Default for new runs |
|---|---|---|
| `checkpoint_layers/lm/aux` | `False/False/False` (set by `recipe.update` in the launch) | `True/True/True` (`PilotArguments`) |
| `proxy_channel_mask` | not supplied, so the receipt is `{'source': 'none; no calibration checkpoint supplied', 'excluded_channels': []}` (inferred from the launch recipe; the saved `train_config.json` was not retrieved) | depends on F1 |

**Consequences:**
- **Checkpoint flags.** If P1/P3 arms run with the defaults, their config
  differs from A's. The report then refuses to compare them, and the arms run
  with different execution settings. The math is the same, but numerics and
  memory differ.
- **Mask.** If the proxy arms use a calibrated mask, their `proxy_mask` receipt
  differs from A's, even though the mask has no effect on arm A (it has no
  targets). The report rejects the comparison.
- **Location.** The baseline lives in its own root
  (`.../proxy-baseline-A-2500-20261005-a01/supervised/run/baseline/seed-42/A`).
  `report()` and `report_seeds()` read `<root>/seed-<n>/<arm>`. The screen queue
  generates arm A again by default, which would retrain it (about 1 h 43 min per
  seed).

**Options:**
- Run every proxy, V and λ = 0 arm with exactly A's execution flags
  (all checkpointing off).
- For the mask, choose one:
  - include arm A in the new screen queue with the same mask flag, so the
    receipts are equal (costs one more A run per seed);
  - make the report ignore `proxy_mask` for arms without a proxy family;
  - use the empty mask for every arm.
- To reuse the existing baseline A, stage its outputs into the screen root, or
  compare it manually as was done for earlier arms.

### F3. Default screen seeds collide with derived initialization streams

**Status:** recommended change before launch.

For the screen, `jobs()` defaults to seeds `[42, 43, 44]` (`deep_kv/__main__.py`,
around line 20). The backbone is initialized with `seed`. The proxy heads use a
separate stream, `seed + 1` (`ProxyModel.__init__`). So:

| Run seed | Backbone stream | Proxy-head stream |
|---|---|---|
| 42 | 42 | **43** |
| 43 | **43** | **44** |
| 44 | **44** | 45 |

The seed-43 run's backbone uses the same random stream as the seed-42 run's
proxy heads, and so on. This is not a correctness bug: within a seed, all arms
stay matched. But it means the "independent" seeds share random streams. The
tensors and shapes differ, so the practical effect is probably negligible. The
same issue was found earlier for the A/B replication, which used seed 123.

**Recommendation:** keep seed 42, since the baseline A already exists, and use
widely separated seeds for the others, for example `[42, 1042, 2042]`. Pass them
explicitly to `make-jobs`.

## 3. Summary

| ID | Finding | Severity | Action |
|---|---|---|---|
| F1 | Mask calibration cannot load the existing `ProxyModel` A checkpoint | Blocking for `M` | Extend the script and add a test, or document the use of an empty mask |
| F2 | Report requires equal configs; baseline A used checkpointing off and no mask | Must decide before the queue | Match A's execution flags; resolve the mask receipt; decide how to reuse A |
| F3 | Seeds 42/43/44 overlap derived `seed + 1` streams | Recommended | Use e.g. 42, 1042, 2042 |

## 4. Resolution (2026-10-05)

F1 is fixed by strict loading of either vanilla wrapper and direct observation
of ProxyModel block outputs. Merely relaxing the old guard was insufficient:
ProxyModel bypasses decoder forward hooks and would silently produce zero
statistics. Calibration now checks per-block token counts and finite values;
unknown checkpoint keys still fail. The calibration output includes token
counts for every block. The saved baseline weights do not need modification.

F2 is fixed by explicit checkpointing=false for all three flags in the screen
recipe, mask-aware report validation, and optional explicit baseline references.
Reports ignore both mask configuration fields for A/V controls while requiring
identical mask paths/receipts for all proxy arms and seeds. Other matching checks
and resume checks remain strict. `make-jobs --reuse-baseline SEED=/path/to/A`
validates the specified A's artifacts/seed/cutoff and skips only its training job;
reports read the original output without copying or rewriting it. New results
must still match the baseline's complete scientific/execution recipe.

F3 is addressed by default seeds 42, 1042, 2042. Backbone/head initialization
algorithms and explicit seed overrides are unchanged, preserving seed 42.
Spacing prevents exact seed-plus-one reuse; spacing itself is not a proof of
statistical independence.

These are local implementation changes only. Calibration on the trained B200
checkpoint and proxy-arm capacity smoke tests still precede a real screen.

Validation: all 21 focused proxy tests passed (31.125 s), followed by all
124 offline CPU tests (138.135 s) in sampling_b200. Regression coverage includes
saved ProxyModel-A restoration, independent per-block activation statistics,
spiked-channel detection, missing-capture rejection, unknown-key rejection,
mask and execution-setting mismatch rejection, cross-seed masks, and explicit
baseline reuse without source changes. The generated reuse queue passed the
runner manifest loader. Evidence: temp/proxy-review-fixes-tests-20261005-a01.log
and temp/proxy-review-fixes-full-20261005-a01.log.
