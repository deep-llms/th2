# Paired log-probability analysis: review and implementation plan

Date: 12 October 2026. Status: review and proposed plan; analysis code and GPU runs are not implemented by this document.

Reviewed inputs:

- [Downstream paired analysis specification, revision 2](downstream_paired_logprob_analysis_spec.md).
- [Reference analysis script](paired_logprob_analysis.py).
- `temp/custom_tasks.zip`: five task definitions and one shared bAbI template.
- Current evaluator: [benchmarks.py](../eval/benchmarks.py), [models.py](../eval/models.py), [suite.py](../eval/suite.py), and [reporting.py](../eval/reporting.py).

## Recommendation

Implement paired likelihood analysis alongside the existing accuracy reports. It can detect changes that do not flip the winning answer, and pairing the same questions between models avoids confounding the comparison with different question samples. Start by analyzing existing few-shot logs, then evaluate the proposed zero-shot suite. No additional pretraining is needed.

The current reference is a useful starting point, but its admission argument, statistical claims, validation guards and precision requirements need revision. The recommendations below intentionally differ from revision 2; a resulting implementation must have its own protocol version and must not claim unchanged revision-2 conformance.

## What the analysis measures

For each example, use exactly the same prompt, choices and gold answer for both checkpoints. Let `ll[c]` be the total log-probability of choice `c`, `n[c]` its character length under the declared task convention, and `g` the correct choice. Use either raw scores `s[c] = ll[c]` or character-normalized scores `s[c] = ll[c] / n[c]`, fixed per task before comparisons.

| Metric | Definition | Interpretation |
|---|---|---|
| M1 | `ll[g] / n[g]` | Likelihood of correct answer text; can improve through general language modeling without improved answer discrimination |
| M2 | `softmax(s)[g]` | Preference for the correct choice relative to alternatives |
| M3 | `s[g] - max(s[wrong])` | Margin against the strongest incorrect choice |
| ACC | `argmax(s) == g` | Original discrete decision under that scoring rule |

M2 from character-normalized scores is a normalized choice score, not a calibrated probability of correctness. WinoGrande also uses different contexts for its alternatives. LAMBADA has M1 and greedy-match accuracy but no M2/M3. Paloma uses document likelihood and UTF-8 byte counts, with separate bits-per-byte reporting.

For M1–M3 and ACC, positive `P6-iso - A` favors P6-iso. For loss and bits per byte, report positive reductions `A - P6-iso`. Preserve ordinary accuracy and effects in natural units alongside standardized effects.

Continuous likelihood metrics have empirical support in [DataDecide](https://arxiv.org/html/2504.11393v1). That supports trying this analysis; it does not guarantee more power for our particular checkpoint pair, nor make these exact metric definitions identical to every metric in that paper.

## Findings that need attention

### 1. Admission using A alone does not establish independence

Revision 2 selects tasks using A-20k's accuracy and A-10k-to-20k improvement, then tests P6-iso against those same A-20k observations. The selection statistic and final difference share A-20k. Not opening P6 results before selection does not remove that dependence.

For intuition, if A and P have equal underlying performance but noisy measurements, selecting cases where A happens to score unusually well tends to make `P - A` negative. This can bias against P6; the issue is valid inference, not the direction of the preferred result. Independence must be justified rather than assumed. See [Kriegeskorte et al. on selection and selective analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/).

**Recommended default:** freeze the full 14-task Suite Z candidate list as the main analysis, and report A's sensitivity/admission table descriptively without filtering that analysis. If admission is essential, select on a disjoint development set and compare on untouched examples, with document/group separation where needed. Keep the unfiltered results visible either way.

Accuracy sensitivity is also not a necessary condition for likelihood sensitivity. A task near chance, or one whose accuracy barely changes with training, can still have informative likelihood differences. Statements such as “cannot resolve” or “changes are noise” should become empirical questions rather than admission axioms.

Existing downstream results have already informed this project. Describe the first study as a planned checkpoint comparison with that history disclosed; reserve stronger independent confirmation for fresh held-out evidence or additional pretraining seeds.

### 2. Sign flipping tests a stronger null than zero average difference

Independent sign flips are justified by exchangeability of the paired outcomes, or appropriate symmetry assumptions on differences. A mean difference of zero alone is insufficient. In particular, positive effects on some tasks and negative effects on others can cancel in the aggregate without satisfying the per-example exchangeability null. [SciPy's permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) describes the paired-exchange null.

The supplied T4 experiment checks independent, symmetric Student-t noise on a fixed task set. It does not validate admission followed by testing, skewed differences, shared-document dependence, or heterogeneous task effects that cancel. The reported T4 percentages were not independently rerun during this review.

Retain paired bootstrap effect estimates and intervals, explicitly state the sign-flip assumptions, and add calibration cases matching these concerns before calling the p-values confirmatory. If the desired hypothesis remains only “aggregate mean effect is zero,” choose and validate inference for that hypothesis rather than claiming finite-sample exactness from exchangeability. A fixed testing order controls multiple testing only when its underlying tests are valid.

Questions/documents are the sampling units for this comparison. These intervals do not capture pretraining-seed variation. Preserve known dependence between questions sharing a source document; distinguish a fixed benchmark inventory from generalization to unseen tasks or BLiMP paradigms.

### 3. Reproduced script issues

| Location | Observed behavior | Required correction |
|---|---|---|
| `check_t5` | Returns true when `n-samples` is absent | Require counts and validate both `original` and `effective` |
| `check_t5` | Returns true when an entire subtask is missing from samples | Compare the complete expected subtask set with results, configs and samples |
| `_std`, `_boot`, `_flip` | A constant nonzero difference is represented as standardized effect zero | Distinguish an all-zero self-comparison from a nonzero effect with zero variance; mark undefined standardized quantities explicitly |
| `task_stats` and callers | Nonfinite inputs can produce numeric p-values instead of stopping | Validate finite scores, positive denominators, sample counts and resampling parameters before statistics; reject nonfinite JSON output |

Concrete local reproductions with `B=100, seed=1`:

- `check_t5({('a', 0): {}}, {})` returned `True`.
- With counts for complete subtasks `a` and `b`, supplying only `a` returned `True`.
- `task_stats(np.ones(20), ...)` returned mean `1`, standardized effect `0`, and `p_std=1`.
- `task_stats(np.array([1., np.nan, 2.]), ...)` returned a NaN mean but a natural-unit p-value of approximately `0.0099`.

Other guards needed during integration: identical targets as well as prompts, checkpoint/tokenizer/backend provenance, complete admission membership if admission is used, and document identity for position-loss arrays. Equal array shapes and NaN masks alone do not establish that rows contain the same documents. Reject empty position buckets or explicitly mark them unavailable; never compute ratios with zero denominators.

### 4. The script is not a direct reader for our artifacts

The reference expects one directory per task containing `results_*.json` and separate `samples_<subtask>_<timestamp>.jsonl`. Our evaluator writes `eval_benchmarks.json`, `eval_benchmarks_full.json`, `eval_samples.jsonl`, and `eval_metadata.json`; each sample has a task field.

Add a small input adapter into the existing evaluation package. Reuse the current model loader, snapshot manifests, task inventory and sequential execution. There is no reason to replace the training code or upgrade the environment merely to read these outputs.

The reference was tested against harness 0.4.13; our pinned dependency is 0.4.10. Match recomputed per-example accuracy and subtask aggregates against actual 0.4.10 outputs. Preserve raw likelihoods and arguments so statistics can be corrected without reevaluating models.

Some protocols also differ from the suite we just implemented:

| Item | Current expanded suite | Proposed paired suite |
|---|---|---|
| MMLU continuation | Five-shot, primary raw accuracy | Zero-shot, character-normalized accuracy |
| SciQ | Zero-shot, primary raw accuracy | Zero-shot, character-normalized accuracy |
| Social IQa | Five-shot, raw accuracy | Zero-shot custom task adding normalized accuracy |
| Paloma | All 16 supported sources | Six specified sources |
| Seeds | Existing evaluator uses 42 for all evaluation RNGs | Suite Z proposes `0,1234,1234,1234`; Suite F retains its historical seeds |

Version and record these changes explicitly. Keep historical scores under their original labels; they are not interchangeable with the new protocol. Continue counting MMLU/BLiMP as one benchmark each, with the declared within-group weighting.

### 5. Precision must distinguish the forward pass from scoring

Our harness adapter already uses `softmax_dtype='float32'`. The model forward can still run with BF16 autocast. Casting BF16 logits to FP32 improves subsequent arithmetic but does not reconstruct an FP32 forward pass.

The current FA4 checkpoint loader requires CUDA BF16, and omitting `--bf16` can still inherit the saved recipe's BF16 setting. The document's generic `dtype=float32` command is therefore not a drop-in command for this wrapper.

Two separate checks are needed:

1. Repeat an identical checkpoint evaluation to measure run-to-run differences.
2. On a fixed, bounded sample, compare the intended production computation with an FP32 SDPA reference at identical weights for both arms. Compare the resulting paired effects as well as individual likelihoods.

An exactly reproducible BF16 calculation can still differ systematically from FP32. The specification's assertion that rounding noise exceeds our effect has not been established by measurement. Freeze tolerances before the study and report measured discrepancies. If the reference changes backend as well as precision, state that this checks their combined effect; use an intermediate SDPA BF16 comparison if attribution matters.

### 6. Custom tasks are plausible; mechanism claims remain tentative

The ZIP contains `social_iqa_norm`, `commonsense_qa_cf`, and bAbI qa1/qa2/qa3. All five loaded under lm_eval 0.4.10 using local fixture data. Target/choice construction worked, and both fixed bAbI demonstrations appeared in each constructed prompt. Evidence: `temp/paired-review-custom-tasks-final.log` (ignored local artifact).

This was a schema/prompt compatibility check, not validation on the real datasets or trained checkpoints. The supplied Social IQa equivalence test, CommonsenseQA choice checks, and bAbI target validation still need real-data coverage.

bAbI hop contrasts and position-dependent losses are useful descriptive probes. Gains at later positions can also reflect context utility or changing token difficulty; gains across bAbI tasks can reflect task-distribution differences. Neither alone proves additional effective depth or a specific internal mechanism.

## Implementation and evaluation plan

| Stage | Work | Completion evidence |
|---|---|---|
| 1. Freeze a revised protocol | Fix task list, scores, shots, RNGs, group weights, inference assumptions and precision checks. Use admission descriptively by default. Keep accuracy and natural-unit effects. | Versioned configuration and explicit differences from revision 2, frozen before new paired results are inspected |
| 2. Implement local analysis | Adapt project artifacts, fix the reproduced guards, preserve pairing and provenance, and add statistics tests. Keep reference inputs as review sources. | Tests for missing tasks/counts, changed targets/prompts, nonfinite values, self-comparison, constant differences and exact small sign-flip cases; calibration under stated assumptions |
| 3. Reuse existing Suite F logs | Locate complete saved per-example outputs for A-20k/P6-iso-20k and validate them against the published aggregate scores. Analyze the eight tasks descriptively. | Exact metric reconstruction and paired identities; if samples are unavailable, record what must be reevaluated rather than infer samples from aggregate scores |
| 4. Add tasks and numerical checks | Integrate the five custom definitions, freeze real input snapshots, check demonstrations/labels, and run bounded numerical and self-comparison checks through the existing model wrapper. | Real-data task audits, finite scores, T1/T2/T5 equivalents, and precision/repeatability reports |
| 5. Evaluate Suite Z | Evaluate A-20k/P6-iso-20k with the frozen zero-shot protocol. Include A-10k as a sensitivity reference if planned, without using its admission results to filter the main comparison. | Complete samples for all 14 tasks, matched prompts and context/truncation audits; task and aggregate reports |
| 6. Run descriptive probes | Evaluate the six Paloma sources and bAbI tasks. Add document-level per-position loss extraction through the existing inference path. | Paloma ratio checks; document IDs/token hashes for position arrays; correctly shifted targets; agreement with training loss on the same documents and boundary policy |
| 7. Report and decide | Report all tasks, averages, natural and standardized effects, uncertainty and exclusions fixed by protocol. Use these results to decide whether another pretraining seed is justified. | Reproducible report with clear checkpoint-level claims and no substitution of significance for practical effect size |

For the position probe, freeze IDs for the first qualifying original validation documents in stored order, not rows from an already packed token cache. Save the tokenized-input hash, truncation policy and checkpoint identity beside each loss array. Compare both arms on those exact inputs. This probe is a new evaluation sample and should not be described as numerically reproducing the original packed-validation PPL.

Future GPU evaluation should use the existing runner's established environment, Accelerate and workload handoff procedures when that execution is requested. This document schedules no workload.

## Reporting requirements

Show the two checkpoint means, paired difference and interval for every metric/task. Include average rows computed from unrounded values with the task weights stated. Keep language-modeling BPB/loss, choice metrics, and fine-tuning scores separate. Do not average standard deviations.

Record source checkpoint hashes, data/tokenizer identities, harness/task versions, seeds, precision/backend settings, context handling, sample counts and validation outcomes. Distinguish observed evidence from assumptions and proposed interpretation. A positive result on this pair is evidence about these checkpoints; it is not yet a result across pretraining seeds or evidence of equal-compute superiority.

The next implementation step is the artifact adapter and statistical validation, followed by reuse of saved Suite F samples. The supplied reference and the existing evaluation code have not been modified as part of this documentation task.
