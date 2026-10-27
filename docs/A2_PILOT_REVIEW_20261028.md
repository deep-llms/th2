# A2 pilot v2.0: review concerns and proposed resolutions

28 October 2026, Singapore time. Review only; no implementation or GPU runs.

Reviewed: [A2_pilot_spec.md](A2_pilot_spec.md), version 2.0.
SHA-256: `3a8b68d1186a5ea201bb79926203017197828518163d701b61789dd005f64f32`.

The pilot asks a useful question: how pretraining length and learning-rate
history affect layer-pruning damage after a fixed healing procedure. Its fixed
layer sets, shared data streams, explicit failure rules and X72 cross-teacher
arm are useful controls. The following concerns should be resolved before
launching the full pilot. They do not establish that the hypothesis is wrong.

These are proposed amendments, not changes to the preregistration. The original
spec is unchanged. Any accepted changes must be versioned before results are
seen; the implementation must not silently substitute these recommendations.

## 1. Outcome-dependent checkpoint eligibility

**References:** T15, section 4.2 (S2), section 6.1; related checks T9 and T10.

T15 rejects S72 when `val(S72) - val(P300) < 0.02`. A checkpoint can be correctly
identified as a stable-LR checkpoint while showing a small annealing gain on
CulturaX. Validation loss is not a direct measurement of its historical LR.
Rejecting it selects the included parents according to an outcome related to
the effect being studied. It also changes the usable-parent pool that selects
S2, affecting Phase 0A's selected pruning set as well.

T9's requirement that loss decrease across position buckets and T10's requirement
that the original RoPE theta give lower loss are empirical expectations, not
implementation invariants. Correct models can fail those expectations on a
particular evaluation sample.

**Proposed resolution:** determine eligibility from checkpoint identity and
documented schedule/iteration metadata. If LR history cannot be established,
mark it unverified. Record the annealing gain, positional profile and alternative
theta losses as diagnostics; investigate surprising values without automatically
excluding a parent because of the direction or size of its loss difference.
Keep the exact RoPE-frequency formula check as a correctness test.

## 2. Final validation also selects hyperparameters

**References:** sections 3.1 and 5.1; the same LR rule applies to Phase 0B.

The grid chooses an LR using final-set loss, then the final set determines the
main decision quantities. In addition, the small set used to select S2 is drawn
from that final set. Thus the final set is not an untouched evaluation set.
LR selection can introduce optimism, potentially different across parents,
which matters when decisions use differences of 0.003–0.008 nats. The magnitude
of this bias has not been measured; this is not a claim that it explains any
result.

**Proposed resolution:** predefine disjoint tuning/calibration and final-scoring
partitions. Select LR and S2 using only the former. Prefer splitting documents
before packing, so chunks of one document do not appear on both sides. Freeze
indices, hashes and the revised sample counts before running the grids. If the
existing layout is retained, explicitly describe the results as development-set
screening rather than independent final evaluation.

The policy that retries a finite run when validation gets worse should also
use the tuning partition if the final partition is to remain untouched.

## 3. Causal attribution needs narrower wording

**References:** sections 4.1 and 6.4(i), (iii), and “How to read the results together.”

X72 and P300 share a teacher, but their parents differ in LR history, post-58k
data exposure and slightly in token count. Their healing LRs are also selected
independently. Therefore `A_init` measures a checkpoint-selection recipe under
the specified LR-selection procedure; it does not isolate LR state alone.
Similarly, S72 versus X72 changes the teacher and may change the chosen LR.

The stable and annealed length comparisons span different token ranges. Dividing
each contrast by its log-length range gives useful descriptive slopes, but
subtracting them does not identify a clean factorial interaction without further
assumptions about the shape of the length effect. The parent used as the KD
teacher also changes with length.

**Proposed resolution:** retain these comparisons, but qualify “only the pruning
point differs,” “only the teacher differs,” and “genuine training-length effect.”
Describe results as evidence under this parent/teacher/healing recipe. Label the
normalized-slope difference descriptive. The public-checkpoint pilot can screen
whether a later controlled experiment is justified; it cannot remove all these
confounds. No additional training arms are requested by this review.

## 4. Export precision may prevent reproduction of decision values

**References:** sections 3.2–3.3 and T4.

Decision evaluation uses FP32 master weights under BF16 autocast, whereas the
saved healed students are BF16. Casting trained master weights to BF16 changes
them. Autocast does not make every operation and parameter use identical to
loading BF16 weights, so the exported checkpoint may yield a different loss.

T4's unhealed round trip does not adequately test this: its original parameters
were already BF16 before upcasting. The discrepancy after healing is currently
unknown, not a demonstrated numerical failure.

**Proposed resolution:** preregister which checkpoint representation determines
the results. Either preserve FP32 decision checkpoints, or make exported BF16
checkpoints the primary evaluation representation. Add a post-healing round-trip
check and record both losses if both representations are used. Do not silently
change the decision precision after observing a discrepancy.

## 5. Screening labels should not imply equivalence or independent replication

**Reference:** section 6.4(i), with implications for both phases' sign counts.

The fallback label “no difference” includes large but inconsistent effects, not
only small effects. Failing to meet a directional threshold is not evidence of
equivalence. Layer sets share parents and evaluation data, so six or twelve
contrasts are not six or twelve independent pretraining replications.

**Proposed resolution:** use “no consistent effect meeting the threshold” or
similarly limited wording. Keep the per-dataset and per-set values visible.
Treat GO/STOP and sign counts as preregistered screening rules, not significance
tests. T14's two reruns provide limited noise information for one condition;
they do not establish uncertainty across all parents, budgets or training seeds.

## 6. Budget verification must include the complete workload

The matrix contains 56 Phase 0A heals, two T14 repeats and 69 Phase 0B heals:
**127 runs before divergence retries**. The stated estimate totals
**22.8 eight-GPU node-hours**, not a measured runtime on this B200 setup.

T8's KD step timing is necessary but insufficient. Re-estimation should include
full and small validation, FP32 Gaussian-sensitivity passes, model loading,
checkpoint saving and setup. Track data preparation/download time separately
and define which costs count toward the phase caps. Preserve the existing
budget gates; this review does not authorize a larger budget or a launch.

## 7. Novelty assessment is incomplete

The referenced `A2_novelty_and_feasibility.md` is absent from this workspace.
The deferred-spec file is also absent; it is not needed to run the current
phases, but its details could not be reviewed.

The broad relationship between optimization history and compression robustness
already has precedents:

- [Training Dynamics Impact Post-Training Quantization Robustness](https://arxiv.org/abs/2510.06213)
  studies the role of LR decay and other training choices in quantization damage.
- [Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://arxiv.org/abs/2003.02389)
  studies weight and LR rewinding for pruning recovery.
- [Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting](https://arxiv.org/abs/2605.02105)
  connects pretraining geometry and annealing to later adaptation and quantization.

A2's candidate contribution is narrower: **layer-pruning recoverability after
healing, examined across pretraining length and annealing state**. That could be
informative, but this review has not established that it is sufficiently novel
for a paper. A positive mechanical GO label would not itself establish novelty.

## Checks actually performed

- Read the complete v2.0 pilot spec and inspected the referenced local filenames.
- Queried public Hugging Face metadata for all 12 pinned parent commits: all
  resolve, with distinct weight hashes within each dataset's four parents.
- Compared the four configs within each dataset: only `_name_or_path` differs.
  The recorded run names agree with the stated batch, context, LR and iterations.
- Compared Hub LFS weight hashes at `iter_0058000`: the 300B and 1T repositories
  match for all three datasets. This establishes identity at that checkpoint;
  it does not independently reconstruct every earlier optimizer/data state.
- Inspected the pinned DCLM 1T main model code. The decoder forward slices by
  `config.num_hidden_layers`, consistent with the proposed removal procedure.
  Attention dropout is copied into each attention module at construction:
  setting only `model.config.attention_dropout` afterwards would be insufficient.

No weight files were downloaded or locally hashed, no parent was instantiated,
and no numerical, throughput or GPU tests were performed. This review is not
a substitute for T1–T15. No source/spec changes, environment modifications,
runner submissions or B200 operations were made.
