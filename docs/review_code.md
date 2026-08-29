# Code Review — Nested Ladder Phase 1 Implementation

Date: 2026-08-29. Scope: the uncommitted working-tree changes implementing
`docs/nested_ladder_design.md` Phase 1 — `compositional/nested_ladder.py`,
`compositional/test_nested_ladder.py`, and modifications to `tied_head.py`,
`loading.py`, `compression_init.py`, `__init__.py`, `train_compositional.py`,
`run_experiments.py`, plus the two new launch scripts and untracked
`scripts/dropbox_downloader.py`. Method: eight independent review angles
(correctness line-scan, cross-file tracer, removed-behavior audit, reuse,
simplification, efficiency, altitude, conventions), every candidate then
adversarially verified against the code; refuted candidates are recorded in
§4 so they are not re-flagged later.

## 1. Overall verdict

**Faithful to the spec and solid at its core.** Verified directly (not from
the status note):

- **All tests pass**: 10/10 in `compositional/test_nested_ladder.py` and
  87/87 across `test_compressed_baselines.py`, `test_tied_head.py`, and
  `tests.py` (env `sparse_emb`, torch 2.7.1). No regressions.
- Tied-head **exactness**, the **budget closed form (18,637,824)**, DDP
  **empty-selection graph connection**, **deterministic structure with zero
  RNG consumption**, and the strict **resume/configless-load round-trip**
  all hold and are covered by tests.
- **Ordering identity between the two claim-1 arms is real**: nested and the
  matched GroupReduce control both rank raw (pseudocount-0) counts through
  the same stable `(-count, token_id)` order; the control launch config
  exactly implements the design's level-set blocks
  (2,048@1024 / 6,144@512 / 24,576@192 / 119,168@64).
- The control being **+785,408 params heavier is intentional** and documented
  in the design doc (in-our-disfavor asymmetry), not a bug.

### Resolution after review

The following findings were fixed on 2026-08-29 and are covered by the
post-review test run: **A1, A2, B1, B2, C2, C3, D1, D2, D4, and D5**. D3's
provenance portion was fixed: the explicit-population GroupReduce path now
logs the counts-file SHA256, resolved populations/ranks, and exact parameter
count. Its proposed "non-descending population" warning was not applied
because the approved matched profile is intentionally ordered by decreasing
frequency and therefore has *increasing* block populations
`2048,6144,24576,119168`; positivity, entry count, and exact sum-to-V are
already enforced.

D7 was partly simplified (unused alternate argument/config forms were
removed), but `V/d/T` and the `member_ids`/`member_slots` convenience
properties remain because §2.4/§2.5 of the approved design explicitly names
that interface. **C1, D6, and D8 remain separate pre-existing work**: do not
launch the TT materialize baseline until C1 is resolved; benchmark/fix the PVQ
repair path before its curriculum; and handle Dropbox-tool behavior as an
operational utility task. None is used by the Nested Ladder or matched
GroupReduce launchers. Post-review validation passes **12/12 Nested Ladder
tests and 109/109 tests across all explicitly selected project suites**, plus
a tiny BF16 forward/backward/AdamW step.

## 2. Findings (most severe first)

Severity groups: **[A] fix before any runner launch on the new th2 node**,
**[B] fix before the flagship runs**, **[C] fix before the affected baseline
/ converter is used**, **[D] polish**. "Pre-existing" marks code untouched by
this diff but load-bearing for planned work.

### A1. Five B200 runner entries monitor the wrong `output_dir` — `run_experiments.py:132,145,158,171,184` (CONFIRMED)
The slim / groupreduce_e2e / tt / global_lowrank / pure_local entries keep
`output_dir` on `OUT_BASE` (default `/opt/dlami/nvme/...`) while their launch
scripts write to `/mnt/local/_outputs/sparse_embedding`. The diff added
`B200_OUT_BASE` to fix exactly this but applied it only to the two new
entries. With `SPARSE_EMB_OUTPUT_BASE` unset (normal on the fresh node),
`--stop-at-step` polls a directory that never exists → the job runs the full
epoch (multi-GPU-days wasted), `require_fresh_output` passes vacuously, and
artifact validation reports success as failure. **Fix:** point all five
entries at `B200_OUT_BASE`.

### A2. Rank allocator hangs when a frequency block is all zeros — `compositional/compression_init.py:149` (CONFIRMED, reproduced)
`allocate_frequency_proportional_ranks` divides by `averages.min()`. The diff
made zero-count block averages reachable (pseudocount guard relaxed to `<0`;
`grouping_pseudocount=0.0` forced on the `--groupreduce_populations` path).
The shipped counts file has 10,398 zero-count tokens; a tail block of only
those makes `ratios` inf/nan and the `while params_at(high) <= target` loop
never terminates — a silent 8-rank wedge at init. Shipped scripts dodge it
only by passing explicit ranks. **Fix:** raise with a clear message when
`averages.min() == 0` (and/or require a positive pseudocount whenever
`target_params` allocation is requested).

### B1. `analyze_anchor_usage.py` crashes on nested checkpoints — `scripts/analyze_anchor_usage.py:65` (CONFIRMED)
`build_embed` calls `_build_arm_from_config(...)` without `state=`; the new
nested branch's state-less path needs counts and raises. Passing
`state=state` (one line) fixes it — and also fixes the pre-existing silent
bug where a PVQ checkpoint is rebuilt with seed-42 *random* assignments
instead of the checkpoint's.

### B2. Nested tied head makes T+1 full N×V logit copies per forward — `compositional/tied_head.py:283` (CONFIRMED)
Out-of-place `index_add` per upper tier plus the out-of-place bias add each
clone the full `(N, V)` logits (~10 GB bf16 at batch 16×2048). Neither op's
backward saves input or output, so in-place `index_add_` / `add_` on the
non-leaf is autograd-safe; saves ~30 GB copy traffic and a ~10 GB transient
per microbatch (~1–2% step time). ⚠️ **Keep the `h·b` bias term** — exact
tying is the repo convention and the exactness tests assert it (one finder
suggested dropping it as a softmax no-op; rejected).

### C1. TT materialization allocates ~123 GB at the production config — `compositional/compressed_baselines.py:638` (CONFIRMED, pre-existing)
`TTEmbedding._lookup` indexes `cores[1][:, digits, :, :]` over the full
padded vocab; at the shipped `tt_tied_r219` config that single intermediate
is 219×160,000×8×219 ≈ 6.1e10 elements ≈ 123 GB bf16 — and the tied head
materializes a second time per step. First step OOMs a 183 GB B200. Unit
tests pass only because they use tiny ranks. **Must be chunked** (vocab
slices, as `TiedMaterializeHead` does) before the TT baseline is scheduled.

### C2. GroupReduce refinement ranks by absolute error, not improvement — `compositional/compression_init.py:326` (CONFIRMED, pre-existing)
`refine_groupreduce_from_dense` filters to improving candidates but spends
the 10% move budget on the *lowest new error* (≈ smallest-norm tokens with
≈0 improvement) instead of the largest `current_error − best_error`,
contradicting its docstring. Weakens the canonical post-hoc conversion that
the allocation-diagnostics gate (`docs/allocation_diagnostics.md`) and the
paper's faithful-GroupReduce baseline rely on. One-line fix to the sort key.

### C3. `pseudocount=0` now fails deep in weighted SVD instead of at parse — `compositional/compression_init.py:33` (CONFIRMED)
The relaxed guard lets `--frequency_pseudocount 0` through
`load_frequency_counts`, but `weighted_low_rank_factors` still requires
strictly positive weights, so the converter dies only after loading the full
dense checkpoint, with a message that no longer names the flag at fault.

### D1. Frequency tie-break rule duplicated across the two arms — `compositional/nested_ladder.py:148` (CONFIRMED)
`_resolve_members` re-implements the stable descending argsort instead of
calling the `frequency_rank_order` helper this same diff added for
GroupReduce. Claim 1 depends on byte-identical rankings; two copies of the
rule is how they silently diverge later. **Fix:** import and call the helper.

### D2. Z/W checkpoint introspection written three times — `compositional/loading.py:147` + `loading.py:411` + `train_compositional.py:~603` (CONFIRMED)
Same sort-keys/read-shapes/rebuild-members block with three different
validation subsets (and ranks read from `W.shape[1]` in one copy,
`Z.shape[1]` in another — equal today, divergent under any shape change).
**Fix:** one `NestedLadderEmbed.structure_from_state(state)` classmethod.

### D3. Matched control lacks frequency-file provenance and population checks — `train_compositional.py:546` (CONFIRMED)
The nested branch logs `file_sha256` of the counts artifact; the groupreduce
populations branch — whose whole purpose is ordering-identity with nested —
logs nothing. Populations are validated only for entry count, not descending
order or budget sanity (a transposed list would launch a ~6× oversized
interface unchallenged). **Fix:** log the sha there too; warn on
non-descending populations.

### D4. Per-depth norm diagnostics add host syncs every microbatch — `train_compositional.py:342` (CONFIRMED)
Masked select + `.sum().item()` per depth per microbatch for metrics emitted
every `logging_steps`; a device-side `zeros(T).index_add_(0, depths−1,
norms)` accumulator read once in `log()` is equivalent. Also couples the
trainer to `NestedLadderEmbed` private attributes via `isinstance` — a
`pop_step_metrics()` protocol on the module would keep trainer edits at zero
for future arms.

### D5. Config-present load path can die with bare `KeyError` — `compositional/loading.py:179` (CONFIRMED)
When `train_config.json` exists, the missing-member-buffer check of the
configless path is bypassed; a state without `member_ids_2` fails as
`KeyError` with no filename or arm context. Reuse the same
`required.issubset(keys)` check on both paths.

### D6. PVQ rebalance loop syncs per candidate on CUDA — `compositional/compression_init.py:416` (PLAUSIBLE, pre-existing)
~4 single-element GPU syncs per surplus candidate inside up to 1000 Lloyd
iterations, on rank 0 while other ranks wait in `broadcast`; could trip the
NCCL watchdog at V=152k/K=1024. Magnitude unverified — benchmark before
launching the PVQ curriculum, or move the greedy repair to CPU lists.

### D7. Dead aliases and unused compatibility branches — `compositional/nested_ladder.py:58` (CONFIRMED)
`self.V/self.d/self.T`, `upper_member_ids()`, the `member_ids`/`member_slots`
properties, the accepts-tier-1-too branch of `_resolve_members`, and
`loading.py`'s `nested_ladder_ranks`/`nested_ladder_populations` fallback
config keys have zero call sites. The `member_ids` property shadowing the
constructor kwarg (while returning only upper tiers) and `self.T` beside
tensor `.T` are misread hazards for the Phase-2 implementer.

### D8. Dropbox tool: single-file `--path` download 409s; `sys.exit` in helpers — `scripts/dropbox_downloader.py:267` (CONFIRMED)
`cmd_download` always calls `files/list_folder` on `--path`, so a file path
returns 409 despite the docstring advertising single-file download; and
`api_request`/`get_token` terminate via `sys.exit(1)`, killing importing
callers on any transient token failure with no retry opportunity.

## 3. Recommended fix order

1. **A1 + A2** before pushing any runner-driven job to the new th2 node.
2. **B1 + B2** before the flagship nested/control launches (both small).
3. **C1** before the TT baseline is ever scheduled; **C2 + C3** before the
   converter-based diagnostics/baselines run.
4. **D1–D8** as polish; D1/D2 are the two that protect scientific validity
   and Phase 2 rather than mere tidiness.

Re-run both test suites after applying (A/B/C changes touch tested paths).

## 4. Candidates examined and refuted (do not re-flag)

- **"Resume breaks when `nested_frequency_path` is relative / missing."**
  Refuted: the resume path uses checkpoint `member_ids_*` and provably never
  touches the frequency file — `test_training_builders_accept_...` asserts
  this with `/does/not/exist`. Fresh runs are guarded by the launch script's
  `test -s`.
- **"The matched control is 4.21% larger — a budget bug."** Intentional and
  documented in `nested_ladder_design.md` §5 (control heavier by exactly
  785,408; in-our-disfavor). The startup log records exact counts per run.
- **"`member_ids=` argument becomes aliased by the module buffer."** Refuted:
  `torch.sort(...).values` allocates a fresh tensor before `register_buffer`.
- **"Drop the tied-head bias as a softmax no-op."** Rejected: exact tying is
  the repo-wide head convention and the fp64 exactness tests assert it.
- **`B200_OUT_BASE` reading the same env var as `OUT_BASE`**: verified
  consistent with both scripts in both env cases; a naming smell only,
  subsumed by A1's real fix.
- **PureLocal/SharedLocal partition-math duplication in `embeddings.py`** and
  the **`analyze_anchor_usage` static-arm-list docstring drift**: real but
  out of this diff's scope (pre-existing, unmodified files, no interaction
  with this change beyond B1).
