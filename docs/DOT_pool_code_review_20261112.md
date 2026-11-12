Closure addendum (13 Nov 2026): the E1-SR diagnostic RNG mismatch described below is now resolved using the two original final checkpoints, without retraining. All four split accuracies match original headlines; corrected route archives and re-entry/CI replay are verified locally. See `results/dot_pool_20261112/final_recovery_a01/FINAL_VERIFICATION.json` and `SR_ERRATUM.json`. Original diagnostic artifacts and this historical review are retained.

# DOT-pool v1.2 — post-run code review, 12 November 2026

Reviewed against `docs/DOT_pool_pilot_spec_v1.2.md` (SHA-256
`a0b41acecbd462db6f9277ebec7173fe242d3493d9fb7f2d77b127d88bc97f05`).
Executed a03 source: `d3ed9a62377eb56c0e065fd5c9ed234b4825484a`.

The review found no demonstrated defect in the executed architecture, training
loss, Taylor gradient scaling, or corrected E2 learning-rate schedule that
requires retraining. This is a bounded review, not a guarantee that every
possible numerical or distributed failure has been excluded. Two concrete
post-training issues and one reporting gap are described below.

## Findings, ordered by practical impact

### P2 — SR route diagnostics use a different evaluation draw from the headline table

Executed `scripts/dot_pool_evaluate.py:17–24` seeds once before traversing all
E1 splits. `efficient_llm/dot_pool/train.py:48–60` resets seed 9182 separately
for every evaluated split. Random routing therefore consumes a different RNG
stream for all route-export splits after ID-dev.

This is confirmed in the already hash-verified local results:

| Run/split | Final evaluation accuracy | Route-export accuracy |
|---|---:|---:|
| E1-SR-s0 ID-test | 0.967 | 0.971 |
| E1-SR-s0 OOD-test | 6/1011 | 5/1011 |
| E1-SR-s1 ID-test | 0.980 | 0.981 |
| E1-SR-s1 OOD-test | 0/1011 | 0/1011 |

The re-entry calculation uses correctness from its own route-export pass, so
its groups are internally consistent. They must not be described as the exact
correct/wrong groups behind `final_eval.json`. This does not invalidate the
training or the headline NLL/accuracy, and does not affect deterministic SO/DOT
inference. SO-s0 and DOT-s0's four exported split accuracies match their final
evaluations exactly in the checked artifacts.

A local fix resets seed per E1 split and allows the route function to use its
model's device for a CPU regression test. The test checks exact SR expert IDs
for all four splits, including a partial last batch. The fix is **not yet
deployed**, and existing route samples/diagnostics are unchanged. To clear the
flag: preserve the old files, export corrected E1-SR routes into a new output,
verify each split accuracy against the original final evaluation, and recompute
the two SR re-entry summaries. No training is required.

More generally, SR uses a stateful random stream: changing batch size, rank
count, or example order changes its draw. E2 headline validation was run with
8 ranks, whereas route diagnostics run on one rank; they should be documented
as separate fixed-seed draws. Batch-invariant random routing was not required
by the spec, and this review does not silently change that algorithm.

### P2 — the original report crashed on duplicate seed fields; recovery remains separately verified

The original report constructed a row using both `seed=<model seed>` and a
diagnostic payload containing `seed=0` for shuffle permutations. Python raised
`TypeError` after all training/downstream/diagnostics had completed.

Commit `a10ac0015bf08d7b02d723ac06c40acdedfdd5c5` separates `seed` from
`diagnostic_seed` and adds the required E1 re-entry CSV and downstream Markdown
table. Full report/recovery fixtures pass. CPU-only additive recovery was
submitted; its retrieved initial snapshot verifies artifacts through E2-PL-s0,
but that snapshot is **not** terminal success. Original failure logs remain.

Do not use `queue_complete.json` or the presence of `REPORT_SMALL.tar.gz` alone
as proof of completion. The queue writes these after a failed report command;
its final wrapper then correctly fails if `report_complete.json` is missing.
This actually happened in a03. Completion must include the report hash,
coverage, replay, archive verification and native handoff, as the recovery does.

### P3 — MoE sanity failure is a reading but is absent from the Flags list

`scripts/dot_pool_report.py:69` emits the E2-R0 label
`MoE implementation or recipe suspect` when the sanity check fails, but does
not append it to `flags`, despite the spec explicitly calling it a flag.
This is a latent reporting defect, not an observed failure of this pilot:
mean D NLL is approximately 3.47611 and mean PL NLL is 3.40881, so PL beats D
by approximately 0.06731 nats and passes the required 0.02 margin.
Before reusing the reporter, add a failed-sanity fixture and propagate the
label to Flags. No current numerical result changes.

## What was checked

Before the local SR fix, all six runtime files (`model.py`, `train.py`,
`data.py`, prepare, queue and evaluate) were byte-identical to executed a03.
Their hashes are saved in the review evidence. No running frozen source was
modified and no GPU command was submitted for this review.

| Area | Review / evidence |
|---|---|
| Architecture | 2 stem + 8/20 middle + 2 coda; E1-L repeats four entire blocks twice; per-step attention, shared/per-layer experts, tied embeddings, QK norm/RoPE and parameter groups match the spec. |
| Expert output | Independent test reconstructs a dense FFN from two half-width experts and verifies the factor `k * normalized_gate`. |
| Ordinary routing | Sigmoid logits in FP32; beta affects selection only; gates exclude beta; distinct top-k; all-reduced loads update beta only after optimizer steps. |
| Balance loss | Independent explicit per-sequence formula and its router gradient match SO, DOT and PL, including excluded PAD and shared-pool aggregation vs mean over PL layers. |
| DOT path | Previous normalized gates and path embeddings feed the router only; gradients remain enabled for the main path. Counterfactual replay uses detached saved inputs/gates and current router/path weights. |
| Taylor labels | Top-C candidates by sigmoid+beta; detached candidate outputs; main-backward hook; summed-task scaling; correct negative sign, population standard deviation and next-step replay. Added test uses nonzero path/depth and uneven microbatches with document masks, checking scores, labels, top candidates and update gradients. |
| Training | AdamW groups/betas/eps, clipping, global valid-target normalization and actual 2000-epoch/1.5B budgets match. Corrected WSD denominator is present in executed a03; the earlier bug was fixed before E2. |
| E1 data | Regenerated seed-2026 KG; checked every answer and ID/OOD edge membership, unique 313600-row train set, all held-out splits disjoint from train, all 1011 OOD bridge queries and one answer target per row. |
| E2 data | Static review of admitted manifest/tokenizer hashes, explicit document boundaries, position resets, cross-document target masks and fixed seed-0 packed-row order. This turn did not re-download or rehash the multi-GB corpus; node admission/health evidence supplies that provenance. |
| Diagnostics | Shuffle preserves current attention/norm/depth while replaying permuted expert IDs/gates; reuse counts selections at least two steps earlier; Taylor validity uses counterfactual-minus-normal summed suffix loss and correlates with its negative. |
| Downstream | Target-token alignment, left truncation to context 512, causal right-padding and canonical request replay reviewed; metrics are reported without selection labels. |

Existing suite: 14 CPU tests passed. Four additional review/regression tests
passed. The full combined run is stored in the evidence directory. Prior node
kernel/production tests remain prior evidence; no fresh B200 numerical test or
distributed CUDA resume test was run in this review. The added Taylor test is
FP32 CPU with fixed label selection, not a proof of every BF16/DDP case.

## Scientific interpretation is separate from implementation correctness

The E1 looped positive control did not develop the registered composition
advantage: mean OOD accuracy is about 0.445%, vs dense about 0.247%.
E1 must retain **“inconclusive: composition did not emerge within budget”**.
The inspected loop implementation is the specified four-block repetition;
failure to grok is not itself evidence of a code bug.

For E2, mean SO minus DOT NLL is approximately 0.001111 nats, below the
registered 0.003 threshold: **“no effect”** for the router on the primary metric.
This should not be promoted to a win merely because a numerical difference is
positive. Time-budget overrun and Tier2/3 node-only backup flags also remain.

Evidence: `results/dot_pool_20261112/code_review_20261112/` contains source
hashes, KG proof, SR artifact comparisons, test logs and verification metadata.
