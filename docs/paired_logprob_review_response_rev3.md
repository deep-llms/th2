# Paired Log-Probability Analysis — Response to Review and Revision 3 Protocol

**Revision 3 (response) — 12 October 2026.** This document responds to `PAIRED_LOGPROB_REVIEW_AND_PLAN_20261012.md`. Together with Revision 2 (`downstream_paired_logprob_analysis_spec.md`), it defines **Revision 3** of the protocol; where the two conflict, this document takes precedence.

- **Reference implementation.** The Revision 2 script is no longer normative. The project implementation becomes the reference once it passes the tests in Section 10.
- **Scope.** This document contains no code or task configs.

## 1. Decisions at a glance

| # | Review finding | Decision | Protocol change |
|---|---|---|---|
| 1 | Selecting tasks on A-20k and then testing A-20k vs P6-iso-20k on the same observations is circular | **Accepted** | The confirmatory set is all 14 Suite Z tasks, with no data-dependent selection. The admission table becomes descriptive. A label-symmetric filter is pre-specified as a secondary analysis (Section 3) |
| 1b | Accuracy sensitivity does not imply likelihood sensitivity | **Accepted** | Revision 2's "cannot resolve" rationale is withdrawn |
| 1c | Earlier results already informed the project | **Accepted** | History is disclosed (Section 3.4), and claims are made at checkpoint level |
| 2 | The sign-flip test assumes more than a zero mean difference | **Accepted, refined** | H0 and its assumptions are restated. Extended calibration covers skewed and cancelling cases. bAbI inference is story-level (Section 4) |
| 3 | Guard bugs in the reference script | **Accepted** | These cases now cause hard failures (Section 5) |
| 4 | Adapter needed; harness 0.4.10; protocol differences | **Accepted** | The project implementation becomes the reference. Protocol choices are frozen in Section 6 |
| 5 | Precision claims were not measured | **Accepted** | Revision 2's bf16 assertion is withdrawn. Repeatability and FP32-equivalence checks with frozen tolerances replace it (Section 8) |
| 6 | Mechanism claims are tentative | **Accepted, extended** | bAbI is clustered by story. A-10k vs A-20k serves as a generic-improvement reference. Wording becomes "consistent with" (Section 9) |
| — | New in this document | **Added** | Effect calibration against A-10k → A-20k (Section 7). Suite F paired results are not opened before the protocol is frozen (T7) |

## 2. Errata to Revision 2

| Revision 2 location | Error | Correction |
|---|---|---|
| §7 Admission rule; §9 confirmatory set | Tasks are selected using A-20k and then tested against the same A-20k observations, so the selection and the test are not independent [4] | Section 3 |
| §7 rationale | "A task that cannot resolve that change cannot resolve 1/56 of it" holds for accuracy, not for likelihood metrics | Withdrawn |
| §9 hypotheses | H1/H2 are stated as $\bar\delta = 0$, but the sign-flip test assumes per-example symmetry | Section 4.1 |
| §8.3 and Appendix A | bAbI examples are treated as independent. In BABILong, consecutive rows are often the same story extended by further sentences, each with its own question [5] | Section 4.3 |
| §4.1 | "Rounding noise in bf16 is larger than the expected effect" was not measured. The project's adapter also already computes the softmax in FP32 | Withdrawn; replaced by Section 8 |
| §3.3, §8.3 | "Supports the depth hypothesis" overstates what a descriptive probe can show | "Consistent with", and only relative to the reference pair (Section 9) |
| Appendix A | `check_t5` passes when `n-samples` is absent or a whole subtask is missing. A constant non-zero difference gives a standardized effect of 0. Non-finite inputs give small p-values (for NaN, $p = 1/(B+1)$) | Section 5 |

## 3. Task set and selection

### 3.1 Confirmatory set

- **Set.** The confirmatory set $C$ is all 14 Suite Z tasks of Revision 2 §3.2.
- **Hypotheses.** H1 (M1) uses all 14 tasks. H2 (M2) uses the 13 multiple-choice tasks; LAMBADA has no M2.
- **Fixed in advance.** No task is added, removed or reweighted after this document.

### 3.2 Admission table

The Revision 2 admission statistics (above-chance SE; paired SE of the 10k → 20k gain) are still computed and reported for every task, as a description of the suite. They do not filter any analysis.

### 3.3 Pre-specified secondary analysis: label-symmetric filter

- **Filter.** Keep the tasks whose pooled accuracy $\bar p_t = (\mathrm{acc}_{A20} + \mathrm{acc}_{P20})/2$ is at least 3 SE above chance, with $\mathrm{SE} = \sqrt{\bar p_t(1-\bar p_t)/N_t}$.
- **Analysis.** Report $\bar\delta$ for M1 and M2 over the kept tasks, with bootstrap CI and sign-flip p-value.
- **Status.** Descriptive. It is not part of the fixed sequence.
- **Why the filter does not bias the test.** $\bar p_t$ does not change when the A and P6-iso outputs of any example are swapped. Conditioning on it therefore leaves the sign-flip null distribution unchanged; this is independent filtering [3].
- **Why A-10k is not used to filter.** A-10k shares A-20k's training trajectory. A filter on A-10k would condition on A-20k's question-level idiosyncrasies again.

### 3.4 History disclosure (to be stated in the report)

- **What was known.** Before this protocol was frozen, the 20k few-shot accuracies, fine-tuning scores and validation PPL of both arms were known. Revision 2's candidate list was drafted with that knowledge: PAWS-X and Belebele were removed.
- **Direction of the removals.** Neither removal favours P6-iso. P6-iso scored higher than A on both (PAWS-X few-shot 50.9 vs 49.85; Belebele zero-shot at 2.5k 23.11 vs 22.89).
- **Scope of claims.** Results from this study concern these checkpoints. Independent confirmation needs a new pretraining seed or another comparison held out from these decisions.

## 4. Inference

### 4.1 Hypotheses and assumptions

For each example $i$ of task $t$, $d_i = M_{P,i} - M_{A,i}$.

| Hypothesis | Null | Statistic | Test |
|---|---|---|---|
| H1 | For every example in $C$, $d_i^{\text{M1}}$ is symmetric about 0 (implied by within-pair exchangeability of the A and P6-iso outputs) | $\bar\delta_{\text{M1}}$ over the 14 tasks | Two-sided sign-flip, $\alpha = 0.05$ |
| H2 (only if H1 is rejected) | The same for $d_i^{\text{M2}}$ | $\bar\delta_{\text{M2}}$ over the 13 multiple-choice tasks | Two-sided sign-flip, $\alpha = 0.05$ |

- **When the test is exact.** The sign-flip test is exact under the stated null.
- **Mean-zero but non-symmetric differences.** If $d_i$ has mean zero but is not symmetric, the test is approximate. The extended calibration (Section 4.4) bounds the error at our sample sizes.
- **Interpreting a rejection.** The checkpoints differ systematically on that metric. The sign of $\bar\delta$ gives the direction, and the bootstrap CI gives the size, in standardized units and in natural units ($\bar D$).
- **Validity of the fixed sequence.** The fixed sequence controls the family-wise error rate at 0.05 on condition that both tests pass the calibration in Section 4.4.

### 4.2 Decision rule

- **Decision.** Use the sign-flip p-value.
- **Report.** Also report the 95% bootstrap CI of $\bar\delta$ and of $\bar D$. If the p-value and the CI disagree about excluding 0, report the result as borderline.
- **What is reported for every task and metric.** Natural-unit effects and accuracy next to the standardized effects.

### 4.3 Sampling units and clustering

| Task | Resampling and flipping unit | Notes |
|---|---|---|
| bAbI qa1–qa3 | **Story** | Two rows belong to the same story if one row's input is a prefix of the other's; stories are the connected components of this relation. Save the cluster IDs. The effective sample size is the number of stories, a few hundred per split; the audit reports the exact count |
| All other tasks | Example | Inference is conditional on the fixed benchmark inventory: the given questions, the 57 MMLU subjects and the 67 BLiMP paradigms. It is not a claim about unseen tasks or paradigms |
| Paloma, position probe | Document | As in Revision 2 |

Simulation (1,000 datasets, $B$ = 1,000; 200 stories × 5 questions, intra-story correlation 0.5, zero effect):

| Procedure | Rejection rate at α = 0.05 | 95% CI coverage |
|---|---:|---:|
| Example-level | 26.2% | 73.1% |
| Story-level | 6.2% | 93.8% |

Example-level inference is therefore invalid for bAbI.

### 4.4 Extended calibration (T4)

The project implementation must reproduce these cases. Thresholds: rejection at α = 0.05 within [3.5%, 6.5%], and coverage within [93.5%, 96.5%] where a true value exists.

| Case | Data | Reference values (1,000 datasets; $B$ = 2,000 for T4a, 1,000 otherwise) |
|---|---|---|
| T4a | Revision 2 case: three tasks, $N$ = 1,000/1,500/2,500, scales 1/0.1/0.02, $t_5$ noise | Zero effect: 5.1% (task), 5.7% ($\bar D$), 5.4% ($\bar\delta$). Coverage with effect: 95.0%, 94.8%, 94.8% |
| T4b | Skewed, mean-zero: centred lognormal(0, 1), $N$ = 500/1,000/1,500, scales 1/0.1/0.02 | 5.2% (task, $N$ = 500), 5.6% ($\bar D$), 5.4% ($\bar\delta$) |
| T4c | Per-task effects that cancel: +0.1, −0.1 and 0 SD, $N$ = 1,000 each, $t_5$ noise | Aggregate rejection 5.4% ($\bar D$), 5.2% ($\bar\delta$) |
| T4d | Story clusters as in Section 4.3 | The story-level procedure must pass; the example-level procedure is expected to fail. Both are reported |
| T4e | Exact small samples, $N \le 12$ | The Monte Carlo sign-flip p-value matches full enumeration of the $2^N$ sign patterns within 3 Monte Carlo SE |

## 5. Implementation guards (hard failures)

| Condition | Required behaviour |
|---|---|
| `n-samples` absent | Fail |
| `original` ≠ `effective` ≠ parsed count | Fail |
| Expected subtask set (from the task or group definition and the results) ≠ subtasks with samples | Fail |
| Example sets, `doc_hash`, `prompt_hash`, `target_hash`, gold index or full arguments differ between arms | Fail |
| Any non-finite log-likelihood, choice length, metric value or difference | Fail |
| Zero denominator: empty choice, zero bytes, zero tokens | Fail |
| $\mathrm{sd}(d) = 0$ with all $d_i = 0$ | $\delta = 0$ (self-comparison) |
| $\mathrm{sd}(d) = 0$ with non-zero constant $d$ | Fail with "standardized effect undefined" |
| Position bucket with no tokens | Mark as unavailable; never compute a ratio from it |
| Position rows | Each row carries a document ID and a token-sequence hash, identical across arms; the truncation policy is recorded |
| Output JSON | Strict: no NaN or Inf |

**Provenance recorded with every result:**

- checkpoint hash, tokenizer hash, harness version (0.4.10) and task-config hashes;
- backend and precision path, batch size and seeds;
- the prefix token used for empty contexts;
- the frozen protocol hash (T7).

## 6. Frozen protocol details

| Item | Revision 3 choice | Note |
|---|---|---|
| Harness | lm-eval 0.4.10 through the project evaluator and an input adapter | Raw log-likelihoods and arguments are kept, so statistics can be recomputed without re-evaluation |
| Suite Z shots | Zero-shot for all 14 tasks | The project's existing 5-shot MMLU-continuation and 5-shot Social IQa runs keep their own labels and are not part of Suite Z |
| Scoring rules | Per-character: ARC-E, ARC-C, HellaSwag, PIQA, SciQ, OpenBookQA, SocialIQA, CommonsenseQA, MMLU-continuation. Raw: WinoGrande, XNLI, XStoryCloze, BLiMP, bAbI. LAMBADA: greedy match plus M1 | Per-character follows the project's own rule ("`acc_norm` where available") and OLMES. Primary raw accuracy for MMLU-continuation and SciQ in the expanded suite departs from that rule. M1 does not depend on this choice |
| Seeds | The evaluator's existing seed (42), recorded | Zero-shot runs sample no demonstrations, and the bAbI demonstrations are fixed. Suite F keeps its historical seeds |
| Paloma | All 16 supported sources | Descriptive. Gating and licences are recorded |
| Groups | MMLU-continuation and BLiMP count as one task each, pooled by example | As in Revision 2 |
| Custom tasks | `social_iqa_norm`, `commonsense_qa_cf`, `babi_cf_qa1`–`qa3` as in Revision 2 | Already loaded under 0.4.10 with fixture data; T6 adds the real-data audits |

## 7. Effect calibration against A-10k → A-20k (new, descriptive)

A-10k is evaluated on Suite Z, the bAbI probe and the position probe. This run is mandatory; its purpose is calibration, not filtering. For each task and for the equal-weight aggregate, report the following for M1, M2 and ACC:

$$
\rho = \frac{\bar D(\text{P6-iso-20k} - \text{A-20k})}{\bar D(\text{A-20k} - \text{A-10k})}
$$

- **CI.** Both differences use the same examples. Compute the CI by resampling examples (stories for bAbI) jointly for numerator and denominator.
- **Unstable cases.** If the denominator's CI includes 0, report $\rho$ as not estimable.
- **Reference value.** The validation-loss ratio is $0.0025 / 0.140 \approx 1.8\%$. A downstream $\rho$ near 1.8% means the downstream gain is what the PPL gain predicts. Values well above or below that are the informative result.

## 8. Precision (replaces Revision 2 §4.1's fp32 requirement)

- **Production path.** bf16 forward with FP32 softmax, as in the current wrapper.
- **Reference path.** FP32 forward with SDPA.
- **Intermediate path.** SDPA bf16, used only if a discrepancy has to be attributed to backend or to precision.

**Frozen subset.** The first 1,000 examples in stored order of HellaSwag, ARC-Easy, PIQA and LAMBADA, plus the first 18 questions of each MMLU subject. Both arms are evaluated on it.

| Check | Statistic | Pass criterion |
|---|---|---|
| Repeatability | Production run twice; $\Delta_i$ = difference in each per-example metric | $\Delta_i = 0$, or $\mathrm{sd}(\Delta^{\text{M1}}) \le 0.01 \cdot \mathrm{sd}(d^{\text{M1}})$ |
| Equivalence: variance | $\Delta_i = d_{i,\text{prod}} - d_{i,\text{ref}}$ for M1 and M2, per task | $\mathrm{sd}(\Delta) \le 0.10 \cdot \mathrm{sd}(d_{\text{ref}})$ |
| Equivalence: bias | Same $\Delta_i$; $n$ = subset size, $N$ = full task size | $\lvert\bar\Delta\rvert + 2\,\mathrm{sd}(\Delta)/\sqrt{n} \le 0.10 \cdot \mathrm{sd}(d_{\text{ref}})/\sqrt{N}$, i.e. bias within 10% of the full-task SE |

- **If every check passes,** use the production path for all suites, including the re-analysis of the saved Suite F logs.
- **If any check fails,** use the reference path for Suite Z. Suite F results stay descriptive, and the measured discrepancy is reported with them.

## 9. Probes

- **bAbI.** Inference is story-level (Section 4.3). For both pairs (P6-iso-20k vs A-20k, and A-20k vs A-10k), report per-task $\delta$ for M1 and M2 and the contrasts $\delta_{\text{qa3}} - \delta_{\text{qa1}}$ and $\delta_{\text{qa2}} - \delta_{\text{qa1}}$.
- **Position.** Freeze the IDs of the first qualifying original validation documents in stored order, not rows of a packed token cache. Save token hashes and the truncation policy. Report $R_b$ for both pairs. When the overall reduction's CI excludes 0, also report the normalized profile $R_b / R_{\text{overall}}$.
- **Interpretation.** A pattern counts as specific to P6-iso only if it differs from the A-10k → A-20k pattern, for example a positive P6-iso contrast with a CI excluding 0 where the reference contrast is not. Results are described as "consistent with" a depth effect. Later-position gains can also reflect context utility or token difficulty, and differences across bAbI tasks can reflect task distribution; the reference pair addresses both only partially.
- **Expected power.** P6-iso's effects are about 1.8% of the reference pair's (Section 7), and bAbI has a few hundred effective units per task. The probes are likely underpowered for P6-iso. Report them regardless.
- **What the position probe is.** It is a new evaluation sample. It does not reproduce the packed-validation PPL.

## 10. Tests

| ID | Test | Change from Revision 2 |
|---|---|---|
| T1 | Parser and metric correctness against actual 0.4.10 artifacts through the adapter | Adds a `target_hash` check |
| T2 | Pairing: example sets, documents, prompts, targets, arguments | Adds targets |
| T3 | Self-comparison on the production path | Unchanged |
| T4 | Calibration T4a–T4e (Section 4.4) | Extended |
| T5 | Completeness, strict (Section 5) | Strict |
| T6 | Custom tasks on real data: bAbI targets are within the six locations; demonstrations; `social_iqa_norm` vs built-in `acc`; five CommonsenseQA choices; bAbI story-cluster audit (number of stories, prefix structure) | Adds the cluster audit |
| T7 | The protocol (Revision 2 plus this document) is frozen and hashed before any paired output (Suite F or Z) is opened | Extended to Suite F |
| T8 | Position losses on the exact frozen inputs agree with the training code's loss on the same inputs within 1e-4 nats; IDs and hashes match across arms | Not compared with the packed-validation PPL |
| T9 | Each Section 5 failure case stops the run | New |
| T10 | Precision checks (Section 8) | New |

## 11. Order of work

The review's plan, merged with the changes above:

1. **Freeze the protocol.** Revision 2 plus this document; record its hash (T7).
2. **Implement and test.** Adapter, guards and statistics; T1–T5 on fixtures, T9 and T4a–T4e.
3. **Precision checks** on the frozen subset, both arms (T10).
4. **Suite F re-analysis** from the saved logs. Descriptive, and only after step 1.
5. **Suite Z** for A-10k, A-20k and P6-iso-20k.
   - Confirmatory H1 → H2 over the 14 tasks.
   - Secondary label-symmetric analysis (Section 3.3).
   - Effect calibration (Section 7).
6. **Probes and Paloma.**
   - bAbI at story level and position probe with frozen IDs, for both pairs.
   - Paloma, all 16 sources.
7. **Report and decide** whether another pretraining seed is justified.

## 12. Retained from Revision 2

- **Unchanged parts.** The following stand as written in Revision 2:
  - metric definitions M1–M3 and ACC;
  - $\bar\delta$ as the confirmatory statistic, with $\bar D$ and accuracy alongside;
  - the fixed sequence H1 → H2;
  - equal task weights;
  - the scoring-rule table;
  - the custom task definitions;
  - the Paloma BPB reduction and the position buckets;
  - the report tables.
- **"Do not average standard deviations."** $\bar\delta$ averages standardized effect sizes, not standard deviations, and is retained. Average rows are computed from unrounded means with stated weights. SDs and SEs are never averaged.

## Sources

[1] `downstream_paired_logprob_analysis_spec.md`, Revision 2 (12 October 2026).

[2] `PAIRED_LOGPROB_REVIEW_AND_PLAN_20261012.md` (project review, 12 October 2026).

[3] Bourgon, Gentleman and Huber, *Independent filtering increases detection power for high-throughput experiments*, PNAS 107 (2010). A filter that is independent of the test statistic under the null leaves the test valid while removing uninformative hypotheses.

[4] Kriegeskorte, Simmons, Bellgowan and Baker, *Circular analysis in systems neuroscience: the dangers of double dipping*, Nature Neuroscience 12 (2009). This is the reference cited in the review.

[5] BABILong, `RMT-team/babilong-1k-samples`, config `0k` (Hugging Face dataset viewer). In `qa1` and `qa3`, the first rows are one story extended by further sentences, each row with its own question.

[6] Gu et al., *OLMES: A Standard for Language Model Evaluations*, arXiv:2406.08446. Per-character normalization is recommended for ARC-Easy, HellaSwag, PIQA, SocialIQA and MMLU in the cloze formulation.
