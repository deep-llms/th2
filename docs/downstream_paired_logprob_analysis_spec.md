# Downstream Evaluation Suite and Paired Log-Probability Analysis — Specification

**Revision 2 — 12 October 2026.** This document specifies which downstream tasks to evaluate, how to run and score them, and how to test whether P6-iso and A differ using **per-example log-likelihoods** and **paired statistics**. It is an implementation contract for the evaluation agent: an implementation that deviates from Sections 3–11 must be labelled as a different analysis. Appendix A contains the reference implementation and Appendix B the custom task configs; both were tested on lm-evaluation-harness 0.4.13 output and on synthetic data.

**Changes from Revision 1**

- Section 3 reviews the current tasks using A's own results. PAWS-X and Belebele are dropped from the zero-shot suite. ARC-C, WinoGrande and XNLI remain candidates but are expected to fail admission.
- New tasks: LAMBADA, SciQ, OpenBookQA, SocialIQA, CommonsenseQA (cloze) and MMLU (cloze), plus BLiMP. Two mechanism probes are added: bAbI with 1–3 supporting facts, and loss by token position. Paloma adds perplexity by domain.
- The confirmatory analysis moves to a zero-shot suite (Suite Z). The reported few-shot suite (Suite F) is analyzed the same way, but only descriptively.
- The confirmatory task set is fixed by an admission rule that uses only A's checkpoints (Section 7).
- The reference implementation now handles task groups, single-continuation tasks, whole-document tasks, admission, Paloma and loss by position. The statistics functions are unchanged from Revision 1.

## 1. Motivation

At 20,000 steps (seed 1042), P6-iso's validation PPL is 0.25% below A's: 17.8416 vs 17.8863, a loss difference of 0.0025 nats/token [1]. Between A-10k and A-20k the loss falls by 0.140 nats, 56 times more. The reported downstream means do not resolve the difference (few-shot: A 49.558, P6-iso 49.441; fine-tuning: A 84.702, P6-iso 84.804) [1]. At A's 10k→20k rate, 0.0025 nats moves even HellaSwag, the most sensitive current task, by only about 0.2 standard errors (SE).

Two consequences follow:

1. **Task choice.** Tasks must carry signal at this scale: above chance, improving with training, and not dominated by noise [8].
2. **Measurement.** Accuracy is replaced by continuous per-example log-likelihood metrics, compared example by example between the two models [2, 3, 4, 5].

## 2. Scope

| Item | Value |
|---|---|
| Confirmatory pair | A-20k vs P6-iso-20k, seed 1042 |
| Admission runs | A-10k and A-20k on Suite Z (Section 7) |
| Descriptive pairs (optional) | A-10.8k vs P6-iso-10k (time-matched at 10k); A-10k vs A-10.8k (sensitivity reference, Section 10); the 15k–19k checkpoints; a time-matched A-20k (≈ 21.6k steps) once trained |

| Suite | Tasks | Shots | Role |
|---|---|---|---|
| Z | 14 candidate tasks (Section 3.2) | 0 | Confirmatory on the admitted tasks, descriptive on the rest |
| F | The reported eight-task suite | HellaSwag 10, ARC-C 25, others 5 | Descriptive; continuity with the reported tables |
| Probes | bAbI qa1, qa2, qa3 (Section 3.3) | 2, fixed | Descriptive mechanism test |
| Perplexity | Paloma, six sources (Section 3.4); loss by token position (Section 3.3) | — | Descriptive |

Out of scope: fine-tuning tasks (Section 3.1) and generation tasks (Section 3.5).

## 3. Benchmark selection

### 3.1 Current tasks: evidence from A

FineWeb2 selects early-signal evaluation tasks by four criteria: improvement as training progresses, low noise, non-random performance early in training, and consistent ordering of models [8]. The first three can be checked on A alone. The table does so on A's reported few-shot accuracies (%) [1]; SE is the binomial standard error at 20k.

| Task | A-20k above chance (SE) | A-10k → 20k (SE) | A-10.8k − A-10k (points) | A at 2.5k* → 10k → 10.8k → 20k | Decision |
|---|---:|---:|---:|---|---|
| HellaSwag | +36 | +10.3 | +0.9 | 29.3 → 37.8 → 38.7 → 42.9 | keep |
| ARC-E | +24 | +4.1 | −0.8 | 36.5 → 45.5 → 44.6 → 49.7 | keep |
| XStoryCloze | +10 | +2.2 | +0.3 | 54.1 → 60.3 → 60.6 → 63.0 | keep |
| PIQA | +17 | +1.8 | +0.2 | 61.2 → 66.4 → 66.5 → 68.3 | keep |
| XNLI | +11 | +0.7 | −2.0 | 38.4 → 43.5 → 41.5 → 44.2 | candidate, expected descriptive |
| ARC-C | +0.4 | +1.4 | +0.6 | 22.8 → 23.7 → 24.3 → 25.5 | candidate, expected descriptive |
| WinoGrande | +2.1 | −0.3 | −1.3 | 51.0 → 53.4 → 52.1 → 53.0 | candidate, expected descriptive |
| PAWS-X | −0.1 | +1.4 (at chance) | +1.3 | 49.5 → 48.3 → 49.6 → 49.9 | dropped from Suite Z |

\* The 2.5k values come from seed 42; all other values come from seed 1042.

- **PAWS-X** is at chance at every checkpoint, so its changes are noise. It remains only in Suite F, for continuity with the reported table.
- **XNLI** is above chance but barely moves between 10k and 20k, and swings by 2 points between two neighbouring checkpoints. Its prompt ("…, right? Yes/Also/No, …") is artificial.
- **ARC-C and WinoGrande** are standard in architecture papers, so they stay in the tables. Their accuracy is within about 2 SE of chance. ARC-C is also noisy between checkpoints at 1B scale [3].
- **Belebele** is dropped. lm-eval scores it by letter choice (A–D) [6], a format in which weaker models stay near chance [9]; A at 2.5k scored 22.9% zero-shot against 25% chance. Its demonstration pool is also the test split [1].
- **Fine-tuning** is not part of this analysis. From A-10k to A-20k, the fine-tuning scores change by only +0.5 to +1.3 points [1]. That is the same order as the spread between fine-tuning seeds (sample SD 0.03–0.75 points at 2.5k) [1], so a 0.0025-nat effect is invisible there.

The binding decision for Suite Z is made by the admission rule on zero-shot runs (Section 7). This table only explains the candidate list.

### 3.2 Suite Z: tasks and scoring

All tasks are scored in **cloze** form: each answer text is scored as a continuation of the prompt. Letter-choice versions (the default `mmlu`, `commonsense_qa` and `belebele` configs) stay near chance for models of this size [9].

| Task | lm-eval task | Split (expected N) | Chance | Scoring rule (reported accuracy) | Status |
|---|---|---|---:|---|---|
| ARC-Easy | `arc_easy` | test (2,376) | ≈ 25% | per-character (`acc_norm`) | current |
| ARC-Challenge | `arc_challenge` | test (1,172) | ≈ 25% | per-character (`acc_norm`) | current |
| HellaSwag | `hellaswag` | validation (10,042) | 25% | per-character (`acc_norm`) | current |
| PIQA | `piqa` | validation (1,838) | 50% | per-character (`acc_norm`) | current |
| WinoGrande | `winogrande` | validation (1,267) | 50% | raw (`acc`) | current |
| XNLI (English) | `xnli_en` | validation (2,490) | 33% | raw (`acc`) | current |
| XStoryCloze (English) | `xstorycloze_en` | eval (1,511) | 50% | raw (`acc`) | current |
| LAMBADA | `lambada_openai` | test (5,153) | — | greedy match (`acc`) | new |
| SciQ | `sciq` | test (1,000) | 25% | per-character (`acc_norm`) | new |
| OpenBookQA | `openbookqa` | test (500) | 25% | per-character (`acc_norm`) | new |
| SocialIQA | `social_iqa_norm` (Appendix B) | validation (1,954) | 33% | per-character (`acc_norm`) | new |
| CommonsenseQA | `commonsense_qa_cf` (Appendix B) | validation (1,221) | 20% | per-character (`acc_norm`) | new |
| MMLU | `mmlu_continuation` (group, 57 subjects) | test (14,042) | 25% | per-character (`acc_norm`) | new |
| BLiMP | `blimp` (group, 67 paradigms) | 67,000 pairs | 50% | raw (`acc`) | new |

"Per-character" divides each choice's log-likelihood by the number of characters in the choice, which is lm-eval's `acc_norm`. "Raw" uses the total log-likelihood, which is `acc`. Each task has one rule, used for ACC, M2 and M3 (Section 6). The expected N is informational; T5 checks counts against the results JSON. Chance is computed from the data as the mean of $1/K_i$ over examples.

**Why these rules.** The current tasks keep the reported protocol: `acc_norm` where the task provides it, otherwise `acc` [1]. The new cloze tasks with answers of varying length use per-character scoring. BLiMP uses raw scoring, which is its standard metric [12]. OLMES recommends PMI normalization for OpenBookQA and CommonsenseQA [9]. This spec uses per-character scoring there too, so that one rule covers every cloze task and no unconditional requests are needed.

**Task notes**

- **LAMBADA.** `lambada_openai` (the OpenAI version of LAMBADA [13]) asks for the last word of a passage. Mamba reports it together with HellaSwag, PIQA, ARC-E, ARC-C and WinoGrande [15]. ACC is lm-eval's `acc`, i.e. whether the word is the greedy continuation. M1 is the per-character log-likelihood of the word. M2 and M3 are undefined because there are no alternatives.
- **SciQ and OpenBookQA.** Use the built-in configs. SciQ's prompt contains the support passage.
- **SocialIQA.** The built-in `social_iqa` logs only `acc`. `social_iqa_norm` (Appendix B) uses the same dataset, prompt and choices and adds `acc_norm`, matching OLMES's per-character normalization for this task [9].
- **CommonsenseQA.** The built-in `commonsense_qa` scores letters. `commonsense_qa_cf` (Appendix B) scores the five answer texts after `"Question: …\nAnswer:"`.
- **MMLU.** `mmlu_continuation` is lm-eval's built-in cloze version. Run it as one invocation. Its 57 subtasks are pooled into one task of 14,042 questions. Each prompt starts with the subject line "The following are questions (with answers) about …".
- **BLiMP.** The `blimp` group has 67 paradigms × 1,000 sentence pairs, pooled into one task. Each pair is scored by the total log-likelihood of the grammatical vs the ungrammatical sentence. The context is empty, so lm-eval's HF wrapper conditions on a single prefix token: `prefix_token_id` from `--model_args` if set, otherwise the tokenizer's BOS token, otherwise EOS (Qwen tokenizers define no BOS) [6]. A model trained with document isolation never sees EOS at position 0, so absolute scores may be slightly depressed; both arms are affected equally. The same applies to XNLI in Suite Z, whose query is empty. Record which token the wrapper uses.
- **Groups.** A group pools all its subtasks with equal weight per example, which equals lm-eval's size-weighted group mean.

### 3.3 Mechanism probes

**bAbI with 1, 2 and 3 supporting facts**

- **Data.** BABILong `RMT-team/babilong-1k-samples`, config `0k`, which contains the original bAbI stories without filler text. Splits `qa1`, `qa2`, `qa3`, about 1,000 stories each [14].
- **Tasks.** `babi_cf_qa1`, `babi_cf_qa2`, `babi_cf_qa3` (Appendix B).
  - Prompt: `"{story}\nQuestion: {question}\nAnswer:"`.
  - Choices: the six bAbI locations (bathroom, bedroom, garden, hallway, kitchen, office), scored by raw log-likelihood. Chance is 1/6.
  - Demonstrations: two fixed, hand-written ones per task, identical for every example and both arms.
- **Built-in tasks not used.** The built-in `babilong` tasks require following an instruction and generating a sentence, which a 0.6B base model does not do.
- **Rationale.** In two-hop queries, the first hop is resolved in early layers and the second in later layers. Models fail when they run out of layers, and patching later-layer states back into earlier layers corrects 32–66% of the failures [10]. P6-iso gives shallow layers an estimate of deeper computation within one pass.
- **Prediction.** If P6-iso adds effective depth, its advantage over A should grow from qa1 (one fact) to qa3 (three facts). Section 8.3 defines the test.
- **Caveat.** If A is near chance on qa2 or qa3, the contrast is uninformative. Report the accuracy levels.

**Loss by token position**

- **Rationale.** P6-iso changes the shallow-layer residual stream, and with it the keys and values that later tokens read. If the gain works through this path, it should grow with position. If it is constant, it comes from each token's own computation.
- **Documents.** The validation documents used for the reported PPL, in stored order: the first 2,000 documents with at least 2,048 tokens, each truncated to 2,048 tokens. If fewer qualify, use all of them and report the number.
- **Per arm.**
  - Run each document alone, at positions 0–2,047 as in training.
  - Compute the fp32 cross-entropy of token $j$ given tokens $< j$, for $j = 1, \dots, 2047$.
  - Save a float32 array of shape [documents, 2048] with column 0 set to NaN.
  - Use the same documents in the same order for both arms.
- **Buckets.** Positions 1–63, 64–127, 128–255, 256–511, 512–1,023 and 1,024–2,047.

### 3.4 Paloma

- **Sources.**
  - Web: `paloma_c4_en`, `paloma_falcon-refinedweb`.
  - Encyclopedic: `paloma_wikitext_103`.
  - Academic: `paloma_m2d2_s2orc_unsplit`.
  - Code: `paloma_dolma_100_programing_languages`.
  - Forums: `paloma_dolma_100_subreddits`.
- **Split.** The test split, which is lm-eval's default.
- **Access.** `allenai/paloma` is gated: accept the AI2 ImpACT license on Hugging Face and use an access token [11]. Some sources carry their own licenses; M2D2, for example, is CC BY-NC.
- **HF wrapper setting.** With an HF-based wrapper, add `logits_cache=False` to `--model_args` for Paloma tasks, as lm-eval's Paloma README instructs [6].
- **Metric.** Bits per byte (BPB) per source. Lower is better.

### 3.5 Tasks not added

- **Generation tasks** (GSM8K, MATH, HumanEval, MBPP, TriviaQA). Their accuracy is near zero at 0.6B parameters and 21B tokens. DataDecide also finds that, unlike the other tasks it studies, GSM8K and Minerva do not become predictable from small runs even with continuous metrics [2].
- **Letter-choice versions of any task** (Section 3.2).

## 4. Running the evaluations

### 4.1 Requirements for every suite

- **Identical settings for both arms.** Harness version, model wrapper, task configs, shots, `--seed`, batch size and dtype are the same. Only the checkpoint differs. T2 verifies that the prompts match.
- **P6-iso is evaluated as trained**, with its proxies active.
- **fp32 scoring** (`dtype=float32` or the wrapper's equivalent). In bf16, logits and log-softmax keep only 8 significant bits. The resulting per-example rounding noise is larger than the expected effect and inflates the variance of $d_i$.
- **Fixed integer batch size per task**, not `auto`. It may differ between tasks but not between arms.
- **One invocation per (checkpoint, task)**, each into its own clean directory. Each directory must contain exactly one `results_*.json`; put re-runs in new directories.
- **Record the harness version**: `lm_eval_version` in the results JSON.

### 4.2 Commands

```bash
# Suite Z (zero-shot); <CKPT> is A-10k, A-20k or P6-iso-20k (plus optional checkpoints)
lm_eval --model <MODEL_CLASS> --model_args <MODEL_ARGS>,dtype=float32 \
  --include_path <CUSTOM_TASKS_DIR> --tasks <TASK> --num_fewshot 0 \
  --batch_size <FIXED_INT> --seed 0,1234,1234,1234 \
  --output_path <OUT>/Z/<CKPT>/<TASK> --log_samples
```

| Run | Change to the command | Output directory |
|---|---|---|
| Probes | `--num_fewshot 2` | `<OUT>/probes/<CKPT>/<TASK>` |
| Suite F | the reported shots and `--seed` | `<OUT>/F/<CKPT>/<TASK>` |
| Paloma | `--tasks paloma_<source>`, add `logits_cache=False` to `--model_args` | `<OUT>/paloma/<CKPT>/<TASK>` |

- `<CUSTOM_TASKS_DIR>` holds the six files of Appendix B.
- `<MODEL_CLASS>` and `<MODEL_ARGS>` are the wrapper and arguments of the reported runs.
- lm-eval's default seed is `0,1234,1234,1234` (Python, NumPy, PyTorch, few-shot sampling). Zero-shot runs do not sample demonstrations; the seed is fixed for uniformity.

### 4.3 Order of work

1. Evaluate A-10k and A-20k on Suite Z.
2. Run `admit` (Appendix A) and save `admission.json`. P6-iso evaluations may run in parallel, but no P6-iso output is opened before this file exists (T7).
3. Evaluate P6-iso-20k on Suite Z. Evaluate both 20k checkpoints on Suite F, the probes and Paloma, and compute their per-position losses (Section 3.3).
4. Run `pair` on Suite Z with `--admission admission.json`; this is the confirmatory result. Then run `pair` on Suite F and on the probes, `paloma`, and `position`; these results are descriptive.

## 5. Sample file fields

lm-eval writes one JSON line per example to `samples_<task>_<timestamp>.jsonl`. The layout below was verified on lm-eval 0.4.13 [6]. Other versions can differ, so inspect the installed version's output before parsing; T1 is the binding check.

**Multiple-choice tasks**

| Quantity | Source |
|---|---|
| Example id | `doc_id` |
| Example and prompt identity | `doc_hash` (SHA-256 of the document) and `prompt_hash` (SHA-256 of the first choice's context) |
| Context and continuation of choice $c$ | `arguments["gen_args_c"]["arg_0"]` and `["arg_1"]`, for $c = 0, \dots, K_i-1$ (sort the keys numerically). Older versions store a list of `[context, continuation]` pairs |
| Log-likelihood $\ell_{i,c}$ | `float(filtered_resps[c][0])`. Each entry is `[log-likelihood, is_greedy]`, stored as strings, e.g. `["-12.34", "False"]` |
| Gold index $g_i$ | `int(target)`; `target` is stored as a string. **WinoGrande:** `target` is the shared continuation text, and the gold index is `{"1": 0, "2": 1}[doc["answer"]]` |
| Choice length $n_{i,c}$ in characters | `len(arg_1) - len(target_delimiter)`. `target_delimiter` is read from `configs[<subtask>]` in the results JSON: one space by default, empty for `paws_en`. This is the length lm-eval uses for `acc_norm` [6] |
| Logged accuracies | `acc`, plus `acc_norm` where the task computes it (0.0 or 1.0) |

**Other formats**

- **Groups** (`mmlu_continuation`, `blimp`). There is one samples file per subtask, `samples_<subtask>_<timestamp>.jsonl`. Delimiters and aggregates are under the subtask's name in `configs` and `results`. Key examples by (subtask, `doc_id`), because `doc_id` restarts in every subtask.
- **LAMBADA** (`loglikelihood`). There is one request: `filtered_resps[0]` is `[log-likelihood, is_greedy]`, and the continuation is a space followed by the last word. The logged `acc` equals `is_greedy`.
- **Paloma** (`loglikelihood_rolling`). `filtered_resps[0]` is the log-likelihood of the whole document, and `target` is the document text. The logged `bits_per_byte` is `[log-likelihood, UTF-8 bytes]`.

Notes:

- ARC questions have 3 to 5 choices; keep per-example arrays.
- WinoGrande places each option in the **context** and scores one shared continuation, so $\ell_{i,c}$ is the log-likelihood of that continuation under context $c$, and all $n_{i,c}$ are equal.
- XNLI and PAWS-X have an empty query. Each choice is the full sentence pair joined by a connector ("Yes", "Also", "No"), so choices are long and share most of their text.

## 6. Per-example metrics

Notation: model $m \in \{A, P\}$ with $P$ = P6-iso; example $i$; choices $c$; gold choice $g$. Each multiple-choice task's **score** follows its scoring rule (Section 3.2):

$$
s_{m,i,c} =
\begin{cases}
\ell_{m,i,c} / n_{i,c} & \text{per-character tasks},\\
\ell_{m,i,c} & \text{raw tasks},
\end{cases}
$$

so $\arg\max_c s_{m,i,c}$ is the choice that the reported accuracy scores.

| ID | Metric | Definition | Measures |
|---|---|---|---|
| **M1** | Gold log-likelihood per character | $\ell_{m,i,g} / n_{i,g}$ | How likely the correct answer text is, on one scale for all tasks. Related to CORRECT PROB [2, §2.5] and bits-per-byte [3]. It can rise with general fluency without better discrimination |
| **M2** | Normalized gold probability | $\exp(s_{m,i,g}) \big/ \sum_c \exp(s_{m,i,c})$ | How strongly the model prefers the correct choice over the alternatives under the task's scoring rule; the continuous counterpart of the reported accuracy. Analogous to NORM CORRECT PROB [2, §2.5] |
| M3 | Margin | $s_{m,i,g} - \max_{c \neq g} s_{m,i,c}$ | Distance from the decision boundary |
| ACC | Reported accuracy | $\mathbb{1}[\arg\max_c s_{m,i,c} = g]$ | Equals the logged `acc_norm` (per-character tasks) or `acc` (raw tasks); T1 checks it |

- **WinoGrande:** $n_{i,g}$ is the length of the shared continuation.
- **LAMBADA:** $\ell_{m,i}$ is the log-likelihood of the last word $w_i$; M1 $= \ell_{m,i} / \mathrm{len}(w_i)$, ACC is the greedy match, and M2 and M3 are undefined.
- **Paloma:** for document $d$, $\ell_{m,d}$ is its total log-likelihood and $b_d$ its UTF-8 byte count; $\mathrm{BPB}_m = -\sum_d \ell_{m,d} \big/ (\ln 2 \sum_d b_d)$.
- **Position:** $L_{m,d,j}$ is the loss of token $j$ of document $d$ (Section 3.3).

All per-example metrics increase with quality. The paired difference for example $i$ is

$$
d_i = M_{P,i} - M_{A,i},
$$

so positive values favour P6-iso.

On per-character tasks the scores of different choices are close, so M2 stays near $1/K_i$ and varies almost linearly with score differences; on raw tasks it is sharper. This is one reason the confirmatory aggregate (Section 8.2) is scale-free.

## 7. Admission rule

The confirmatory task set $C$ is chosen from Suite Z with **A's checkpoints only**, before any P6-iso result is seen. For each task $t$, with ACC computed under the task's scoring rule and subtasks pooled:

| Criterion | Definition | Threshold |
|---|---|---|
| Non-random | $(\mathrm{acc}_{20} - c_t) / \mathrm{SE}_{20}$, with chance $c_t$ = mean of $1/K_i$ (0 for LAMBADA) and $\mathrm{SE}_{20} = \sqrt{\mathrm{acc}_{20}(1-\mathrm{acc}_{20})/N_t}$ | $\ge 3$ |
| Sensitive | $\bar e / (\mathrm{sd}(e)/\sqrt{N_t})$, with $e_i = \mathrm{ACC}_{20,i} - \mathrm{ACC}_{10,i}$ on the same examples | $\ge 2$ |

- A task is **admitted** if it meets both criteria. $C$ is the set of admitted tasks. All other Suite Z tasks are reported descriptively.
- **Why sensitivity.** From A-10k to A-20k the loss falls by 0.140 nats. A task that cannot resolve that change cannot resolve 1/56 of it.
- **Why selection does not bias the comparison.** Selection uses only A's level and A's improvement, not any difference between A and P6-iso.
- **Too few tasks.** If fewer than three tasks are admitted, there is no confirmatory test and every result is descriptive.
- **SE in Section 3.1.** The table in Section 3.1 uses the single-run SE. The binding rule uses the paired SE of the gain, which is smaller when the two checkpoints agree on most examples.
- The rule, its thresholds and the candidate list are fixed by this document and are not changed after any result is seen.

## 8. Statistics

### 8.1 Per task (descriptive)

For task $t$ with $N_t$ examples:

| Statistic | Definition |
|---|---|
| Mean difference | $\bar d_t = \frac{1}{N_t} \sum_i d_i$ |
| Standardized effect | $\delta_t = \bar d_t / \mathrm{sd}(d)$, with the sample standard deviation ($N_t - 1$) |
| 95% CI | Paired bootstrap with $B = 10{,}000$: resample examples with replacement, recompute $\bar d_t$ and $\delta_t$, and take the 2.5th and 97.5th percentiles |
| p-value | Sign-flip test with $B = 10{,}000$: multiply each $d_i$ independently by $\pm 1$, recompute the statistic $T_b$, and apply the formula below |
| Direction counts | Fractions of examples with $d_i > 0$ and with $d_i < 0$ |

$$
p = \frac{1 + \#\{b : \lvert T_b \rvert \ge \lvert T_{\text{obs}} \rvert\}}{B + 1}
$$

Swapping the two models' outputs on one example flips the sign of its $d_i$. The sign-flip test is therefore an exact paired permutation test of the null hypothesis that the two models are exchangeable on every example.

With many tasks and four metrics, some per-task results will look significant by chance. No claim is made from an individual task.

### 8.2 Aggregate over tasks

Tasks are weighted equally, as in the reported "Average" rows. Two aggregates are computed for every metric, first over $C$ (confirmatory) and then over all Suite Z tasks (descriptive):

| Aggregate | Definition | Role |
|---|---|---|
| Standardized | $\bar\delta = \frac{1}{T} \sum_t \delta_t$ | **Confirmatory statistic (Section 9)** |
| Natural units | $\bar D = \frac{1}{T} \sum_t \bar d_t$ | Descriptive. For ACC, $100\,\bar D$ equals the difference between the two arms' "Average" rows |

- **Which tasks enter which aggregate.** The M1 and ACC aggregates include LAMBADA. The M2 and M3 aggregates include only multiple-choice tasks.
- **CI:** stratified bootstrap. Each replicate resamples examples independently within each task, computes each task's statistic, and averages over tasks.
- **p-value:** sign-flip test with independent flips for every example of every task, averaging the per-task statistics of each flip set.

**Why the confirmatory statistic is standardized.** The scale of $d_i$ can differ between tasks by an order of magnitude (number and length of choices, per-character or raw scores), so $\bar D$ is dominated by the tasks with the largest scale. A synthetic check used the T4 setup: three tasks with scales 1, 0.1 and 0.02, a standardized effect of 0.05 in each, and 1,000 simulations. The test on $\bar\delta$ detected the effect in 90% of simulations and the test on $\bar D$ in 39%. Under zero effect, both had false-positive rates close to 5% (5.4% and 5.7%).

### 8.3 bAbI hop contrasts (descriptive)

For M1 and M2, report $\delta_{\text{qa2}} - \delta_{\text{qa1}}$ and $\delta_{\text{qa3}} - \delta_{\text{qa1}}$. Each 95% CI comes from independent bootstrap resamples of the two tasks, whose stories are different. A positive contrast whose CI excludes 0 supports the depth hypothesis of Section 3.3. In 500 simulations, the CI covered the true contrast in 94.6% of cases.

### 8.4 Paloma (descriptive)

For each source, the BPB reduction

$$
R = \mathrm{BPB}_A - \mathrm{BPB}_P = \frac{\sum_d (\ell_{P,d} - \ell_{A,d})}{\ln 2 \, \sum_d b_d}
$$

is positive when P6-iso is better. Its 95% CI comes from a bootstrap over documents (pairs resampled together), and its p-value from a sign-flip test over documents. In 500 simulations, this ratio statistic had a 4.6% false-positive rate and 95.0–95.2% coverage.

### 8.5 Loss by token position (descriptive)

For bucket $b$, the loss reduction in nats/token is

$$
R_b = \frac{\sum_d \sum_{j \in b} (L_{A,d,j} - L_{P,d,j})}{\sum_d n_{d,b}},
$$

where $n_{d,b}$ is the number of tokens of document $d$ in bucket $b$. Positive values favour P6-iso. CIs and p-values use the document bootstrap and sign-flip test of Section 8.4. The contrast $R_{1024\text{–}2047} - R_{1\text{–}63}$ gets a document-bootstrap CI.

## 9. Confirmatory test

Two hypotheses are tested in a fixed order (fixed-sequence procedure [7]), each two-sided at $\alpha = 0.05$, using the sign-flip p-value of $\bar\delta$ over the admitted set $C$:

1. **H1:** $\bar\delta_{\text{M1}} = 0$ over $C$ (likelihood of the correct answers).
2. **H2:** $\bar\delta_{\text{M2}} = 0$ over the multiple-choice tasks in $C$ (discrimination of the correct answer). Tested **only if H1 is rejected**, and only if $C$ contains at least three multiple-choice tasks.

This order keeps the family-wise error rate at 0.05 without adjusting α. The bootstrap CIs report effect sizes; the decision uses the p-values.

| Outcome | Allowed statement |
|---|---|
| H1 not rejected | No detectable difference in the likelihood of correct answers. M2 is reported, not tested |
| H1 rejected, H2 not rejected | P6-iso assigns higher (or lower, by the sign of $\bar\delta_{\text{M1}}$) likelihood to correct answers; no detectable difference in discrimination |
| H1 and H2 rejected | As above, and P6-iso discriminates the correct answer better (or worse, by the sign of $\bar\delta_{\text{M2}}$) |

Only the last outcome, with positive signs, supports the statement that P6-iso's PPL gain carries over to downstream tasks. If a hypothesis is not rejected, report its CI: the CI bounds the effect sizes that the data still allow.

The metrics, scoring rules, admission rule, aggregate, test order and α are fixed by this document before any result is seen and are not changed afterwards. M3, ACC and every analysis outside Suite Z are descriptive.

## 10. Interpretation limits and optional analyses

- **One pretraining seed per arm.** Questions are the random unit, so the test says whether these two checkpoints differ on this population of questions, not whether the two methods differ across pretraining seeds. When a second A seed at 20k exists, analyze A (seed 1042) vs A (second seed) the same way. P6-iso's difference is informative about the method only if it clearly exceeds that seed-to-seed difference.
- **Sensitivity reference.** A-10k vs A-10.8k differs by 0.0161 nats of validation loss (PPL 20.5687 vs 20.2394) [1], about 6.5 times the 20k effect. If Suite Z does not resolve this pair on M1, it will not resolve the 20k pair. The converse does not hold, because two A runs with the same seed may differ less per example than A and P6-iso do.
- **Checkpoint consistency.** Analyze the 15k–19k checkpoint pairs and report $\bar\delta$ per step. Checkpoints of one run are correlated, so consistency supports but does not replace a second seed.
- **Late-checkpoint average.** Average each example's metric over the 16k–20k checkpoints of each arm, then apply Sections 8.1–8.2. This reduces checkpoint-to-checkpoint noise [3] but measures the late-training average rather than the final checkpoint.
- **Descriptive analyses.** Suite F, the probes, Paloma and loss by position are descriptive. A positive result there supports, but does not replace, the confirmatory result.

## 11. Integration tests

All must pass before any result is reported.

| ID | Test | Pass criterion |
|---|---|---|
| T1 | Parser and metric correctness, both arms | The task's reported accuracy key is logged for every example, and every check below holds for every subtask. **Multiple choice:** `acc` and `acc_norm` recomputed from the parsed $\ell$, $n$ and $g$ equal the logged per-example values exactly, and their means equal `acc,none` and `acc_norm,none` (within 1e-12). WinoGrande logs only `acc`. **LAMBADA:** `acc` equals `is_greedy`, and $\exp(-\text{mean}\,\ell)$ equals `perplexity,none`. **Paloma:** the byte counts equal the logged ones, and the recomputed BPB equals `bits_per_byte,none`. Floating-point comparisons for LAMBADA and Paloma use a relative tolerance of 1e-9 |
| T2 | Pairing | Identical example sets, and identical `doc_hash`, `prompt_hash` and full `arguments` (all contexts and continuations) for every example. Any mismatch means the prompts differ: stop |
| T3 | Self-comparison | Evaluate one checkpoint twice in separate invocations and analyze the pair. Every $d_i$ should be exactly 0. Otherwise report $\max_i \lvert d_i \rvert$ for M1, which must be below 1% of $\mathrm{sd}(d)$ for M1 in the A vs P6-iso comparison, for every task |
| T4 | Statistical calibration on synthetic data | Three tasks with $N$ = 1,000 / 1,500 / 2,500, scales 1 / 0.1 / 0.02 and unit-variance $t_5$ noise; $B$ = 2,000; 1,000 simulations. With zero effect, the rejection rate at α = 0.05 lies in [3.5%, 6.5%] for the per-task, $\bar D$ and $\bar\delta$ tests. With a standardized effect of 0.05 in every task, the 95% CIs contain the true per-task mean, $\bar D$ and $\bar\delta$ in [93.5%, 96.5%] of simulations |
| T5 | Completeness | For both arms and every subtask, the number of examples equals `n-samples` in the results JSON (`original` = `effective`) and the count of the reported runs where one exists; no duplicate examples |
| T6 | Custom tasks | **bAbI:** every target is one of the six locations (the config raises an error otherwise), and both demonstrations appear verbatim at the start of every prompt. **SocialIQA:** `acc` of `social_iqa_norm` equals `acc` of the built-in `social_iqa` on A-20k (run both once). **CommonsenseQA:** every question in `commonsense_qa_cf` has five choices |
| T7 | Admission before pairing | `admission.json` is written before any P6-iso output is opened, and its admitted list is copied into the report unchanged |
| T8 | Position losses | For each arm, the token-weighted mean of the loss array equals the loss that the training code reports for the same documents, within 1e-4 nats |

With the reference implementation (Appendix A), T4 gives the following values. Under zero effect, the rejection rates are 5.1% (per task), 5.7% ($\bar D$) and 5.4% ($\bar\delta$). With an effect, coverage is 95.0%, 94.8% and 94.8% respectively. The run takes about 5 minutes on a small CPU machine.

If a test fails, stop and report the failure with its numbers. Tests and metrics are not changed to make a result pass.

## 12. Report

1. **Admission table.** For every Suite Z task: N, chance, ACC at 10k and 20k, the non-random and sensitivity statistics, and the decision.
2. **Confirmatory result.** $\bar\delta$ for M1 and M2 over $C$, with 95% CI and sign-flip p-value, and the outcome of Section 9.
3. **Suite Z aggregates**, over $C$ and over all tasks. For M1, M2, M3 and ACC: $\bar D$ and $\bar\delta$, each with CI and p-value, plus each arm's "Average" row.
4. **Suite Z per task.** For each task and metric: $N_t$, the A and P6-iso means, $\bar d_t$ with CI and p-value, $\delta_t$, and the fractions of examples with $d_i > 0$ and $d_i < 0$.
5. **Suite F**, in the same format as items 3 and 4 (descriptive).
6. **Probes.** For each bAbI task: ACC, M1 and M2 for both arms, the paired differences with CIs, and the hop contrasts.
7. **Paloma.** For each source: BPB of each arm and $R$ with CI and p-value.
8. **Position.** For each bucket: $R_b$ with CI, plus the last-minus-first contrast.

**Required statements:**

- the lm-eval version, model wrapper, dtype, batch sizes and seeds (few-shot, bootstrap, sign-flip);
- the prefix token used for empty contexts (BLiMP and zero-shot XNLI);
- the outcomes of T1–T8;
- the new runs' accuracies next to the reported tables;
- that per-task and non-Suite-Z results are descriptive;
- that each arm has one pretraining seed (Section 10).

## 13. Defaults

| Parameter | Value |
|---|---|
| Confirmatory suite | Suite Z, zero-shot, admitted tasks $C$ |
| Admission thresholds | Non-random $\ge 3$ SE; sensitivity $\ge 2$ paired SE; at least three admitted tasks |
| Confirmatory hypotheses | M1, then M2 (fixed sequence), two-sided, α = 0.05 |
| Confirmatory statistic | $\bar\delta$, the equal-weight mean of per-task standardized effects |
| Decision | Sign-flip p-value |
| Replicates $B$ | 10,000 for every bootstrap and sign-flip test |
| Confidence intervals | 95%, percentile |
| Task weights | Equal; groups pool their subtasks |
| Scoring dtype | fp32 (bf16 only if necessary, identical for both arms) |
| Batch size | Fixed integer per task, identical for both arms |
| Probes | bAbI qa1–qa3, two fixed demonstrations, raw scoring |
| Paloma sources | c4_en, falcon-refinedweb, wikitext_103, m2d2_s2orc_unsplit, dolma_100_programing_languages, dolma_100_subreddits |
| Position buckets | 1–63, 64–127, 128–255, 256–511, 512–1,023, 1,024–2,047 |
| Analysis seed | 20261012; per-task statistics use 20261012 + k (k = task index) and aggregates 20261012 + 1000 + j (j = metric index) |

## Appendix A. Reference implementation

`paired_logprob_analysis.py` (A.2, also provided as a separate file) implements Sections 5–9 and the checks T1, T2 and T5. T3 is a run of `pair` on two evaluations of one checkpoint. T4 is `t4()` in A.3. T6–T8 are manual checks. Pass A first in `pair`, `paloma` and `position`, because differences are computed as B − A.

### A.1 Usage and testing

```bash
S=paired_logprob_analysis.py
python $S admit --a10 <OUT>/Z/A-10k --a20 <OUT>/Z/A-20k --out admission.json
python $S pair <OUT>/Z/A-20k <OUT>/Z/P6-iso-20k --admission admission.json --out Z_pair.json
python $S pair <OUT>/F/A-20k <OUT>/F/P6-iso-20k --out F_pair.json \
  --tasks arc_challenge arc_easy hellaswag piqa paws_en winogrande xnli_en xstorycloze_en
python $S pair <OUT>/probes/A-20k <OUT>/probes/P6-iso-20k --out probes_pair.json \
  --tasks babi_cf_qa1 babi_cf_qa2 babi_cf_qa3
python $S paloma <OUT>/paloma/A-20k <OUT>/paloma/P6-iso-20k --out paloma.json
python $S position loss_A-20k.npy loss_P6-iso-20k.npy --out position.json
```

Each command stops if T1, T2 or T5 fails. Each prints a summary and writes everything to its JSON file. `Z_pair.json` contains `confirmatory`, the outcome of Section 9.

**What was tested.** The tests used lm-eval 0.4.13 output from its `dummy` model, which returns random log-likelihoods. They used toy tasks in the format of every task type here, and the Appendix B configs with local copies of their dataset schemas; Hugging Face was not reachable from the test environment. The covered task types were:

- multiple choice with per-character and raw scoring, and with a shared continuation;
- a two-subtask group with a description line (MMLU style) and an empty-context group (BLiMP style);
- a single-continuation task (the LAMBADA config);
- a whole-document task (Paloma style, including non-ASCII text);
- the three bAbI configs, `commonsense_qa_cf` and `social_iqa_norm`.

Results:

- T1, T2 and T5 pass on matched runs of every task type.
- A run with different few-shot settings fails T2, and a task whose reported key is not logged fails T1.
- A self-comparison gives $d_i = 0$ and $p = 1$ everywhere.
- The older list format of `arguments` and numeric responses gives identical results.
- Admission admits nothing for random-model runs. For a run with a real improvement on three tasks, it admits all three, and the confirmatory test then rejects H1 and H2.
- The Paloma reduction equals $\mathrm{BPB}_A - \mathrm{BPB}_B$, and the recomputed BPB equals lm-eval's `bits_per_byte`.
- The position analysis recovers a planted 0.01-nat gain at positions ≥ 1,024.
- The statistics functions of Revision 1 are byte-identical, so the T4 values still hold. The calibration figures for the new ratio and contrast statistics are given in Sections 8.3 and 8.4.

The script has not yet been run on real checkpoints or on the real datasets.

### A.2 Analysis script

```python
#!/usr/bin/env python3
"""Paired per-example analysis of lm-eval `--log_samples` runs.

Reference implementation for downstream_paired_logprob_analysis_spec.md (Revision 2).
Differences are always B - A (positive values favour B). Subcommands:
  admit    --a10 DIR --a20 DIR [--tasks ...] --out admission.json     task admission from A only (Section 7)
  pair     DIR_A DIR_B [--tasks ...] [--admission admission.json]       paired analysis (Sections 6, 8, 9)
  paloma   DIR_A DIR_B [--tasks ...]                                    bits-per-byte by source (Section 8.4)
  position LOSS_A.npy LOSS_B.npy                                        loss by token position (Section 8.5)
A run directory holds one subdirectory per task: RUN_DIR/<task>/**/results_*.json and samples_*.jsonl
(a group task such as mmlu_continuation has one samples file per subtask; all subtasks are pooled).
"""
import argparse, glob, json, math, os, re
import numpy as np

# task -> (kind, scoring rule, reported accuracy key). kind: "mc" multiple choice, "ll" one continuation.
TASKS = {
    "arc_easy": ("mc", "char", "acc_norm"), "arc_challenge": ("mc", "char", "acc_norm"),
    "hellaswag": ("mc", "char", "acc_norm"), "piqa": ("mc", "char", "acc_norm"),
    "sciq": ("mc", "char", "acc_norm"), "openbookqa": ("mc", "char", "acc_norm"),
    "social_iqa_norm": ("mc", "char", "acc_norm"), "commonsense_qa_cf": ("mc", "char", "acc_norm"),
    "mmlu_continuation": ("mc", "char", "acc_norm"),
    "winogrande": ("mc", "raw", "acc"), "xnli_en": ("mc", "raw", "acc"), "xstorycloze_en": ("mc", "raw", "acc"),
    "paws_en": ("mc", "raw", "acc"), "blimp": ("mc", "raw", "acc"),
    "babi_cf_qa1": ("mc", "raw", "acc"), "babi_cf_qa2": ("mc", "raw", "acc"), "babi_cf_qa3": ("mc", "raw", "acc"),
    "lambada_openai": ("ll", None, "acc"),
}
SUITE_Z = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "xnli_en", "xstorycloze_en",
           "lambada_openai", "sciq", "openbookqa", "social_iqa_norm", "commonsense_qa_cf",
           "mmlu_continuation", "blimp"]
PROBES = ["babi_cf_qa1", "babi_cf_qa2", "babi_cf_qa3"]
PALOMA = ["paloma_c4_en", "paloma_falcon-refinedweb", "paloma_wikitext_103", "paloma_m2d2_s2orc_unsplit",
          "paloma_dolma_100_programing_languages", "paloma_dolma_100_subreddits"]
# Subtasks whose choices sit in the context and share one continuation; gold index as in the task's doc_to_text.
MULTI_INPUT_GOLD = {"winogrande": lambda doc: {"1": 0, "2": 1}[doc["answer"]]}
METRICS = ("M1", "M2", "M3", "ACC")
SAMPLES_RE = re.compile(r"samples_(.+)_(\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")


def one_file(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one file for {pattern}, found {len(files)}")
    return files[0]


def _args(s):
    args = s["arguments"]
    if isinstance(args, dict):  # lm-eval >= 0.4.3: {"gen_args_N": {"arg_0": ..., "arg_1": ...}}
        args = [args[k] for k in sorted(args, key=lambda k: int(k.rsplit("_", 1)[1]))]
        args = [[a[f"arg_{j}"] for j in range(len(a))] for a in args]
    return [tuple(a) for a in args]


def _parse(s, kind, sub, delim, rule):
    args = _args(s)
    r = {"args": args, "doc_hash": s["doc_hash"], "prompt_hash": s["prompt_hash"],
         "logged": {k: s[k] for k in ("acc", "acc_norm", "bits_per_byte") if k in s}}
    resp = s["filtered_resps"]
    if kind == "rolling":                                   # whole document: ll and UTF-8 bytes
        r["ll"] = float(resp[0] if isinstance(resp, list) else resp)
        r["bytes"] = len(s["target"].encode("utf-8"))
        return r
    if kind == "ll":                                        # one continuation (LAMBADA)
        cont = args[0][1]
        r.update(ll=float(resp[0][0]), greedy=str(resp[0][1]) == "True", n_ans=float(len(cont.strip())))
        return r
    cont = [a[1] for a in args]
    if not all(c.startswith(delim) for c in cont):
        raise RuntimeError(f"{sub} doc {s['doc_id']}: continuation lacks the target delimiter")
    r["ll"] = np.array([float(x[0]) for x in resp], dtype=np.float64)
    r["n"] = np.array([len(c) - len(delim) for c in cont], dtype=np.float64)  # = len(choice), as acc_norm
    r["multi"] = sub in MULTI_INPUT_GOLD
    if r["multi"]:
        if rule == "char" or len(set(cont)) != 1 or cont[0] != delim + s["target"]:
            raise RuntimeError(f"{sub} doc {s['doc_id']}: not a shared-continuation task scored raw")
        r["gold"], r["n_ans"] = MULTI_INPUT_GOLD[sub](s["doc"]), r["n"][0]
    else:
        r["gold"] = int(s["target"])
        r["n_ans"] = r["n"][r["gold"]]
    return r


def load_task(run_dir, task, kind=None):
    """Parse one run's output for one task. Returns ({(subtask, doc_id): record}, results JSON)."""
    kind = kind or TASKS[task][0]
    rule = TASKS[task][1] if task in TASKS else None
    tdir = os.path.join(run_dir, task)
    with open(one_file(os.path.join(tdir, "**", "results_*.json")), encoding="utf-8") as f:
        results = json.load(f)
    files = sorted(glob.glob(os.path.join(tdir, "**", "samples_*.jsonl"), recursive=True))
    if not files:
        raise RuntimeError(f"{task}: no samples files under {tdir}")
    recs = {}
    for fn in files:
        sub = SAMPLES_RE.search(os.path.basename(fn)).group(1)
        delim = results["configs"][sub].get("target_delimiter")
        delim = " " if delim is None else delim
        with open(fn, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                key = (sub, s["doc_id"])
                if key in recs:
                    raise RuntimeError(f"{task}: duplicate example {key}")
                recs[key] = _parse(s, kind, sub, delim, rule)
    return recs, results


def example_metrics(r, kind, rule):
    if kind == "ll":
        return {"M1": r["ll"] / r["n_ans"], "M2": np.nan, "M3": np.nan, "ACC": float(r["greedy"])}
    ll, g = r["ll"], r["gold"]
    s = ll / r["n"] if rule == "char" else ll               # scores of the reported accuracy
    p = np.exp(s - s.max())
    p /= p.sum()
    return {"M1": ll[g] / r["n_ans"],                       # per-character log-likelihood of the gold answer
            "M2": p[g],                                     # normalized probability of the gold answer
            "M3": s[g] - np.delete(s, g).max(),             # margin over the best wrong choice
            "ACC": float(np.argmax(s) == g)}                # reported accuracy (acc_norm or acc)


# ---------------------------------------------------------------- integration tests T1, T2, T5
def _close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(b))


def check_t1(recs, results, task, kind=None):
    """Recomputed per-example metrics equal the logged ones; subtask means equal the results JSON."""
    kind = kind or TASKS[task][0]
    key = TASKS[task][2] if task in TASKS else None
    bad, by_sub = 0, {}
    for (sub, _), r in recs.items():
        by_sub.setdefault(sub, []).append(r)
    for sub, rs in by_sub.items():
        agg = results["results"][sub]
        if kind == "rolling":
            ll, nb = sum(r["ll"] for r in rs), sum(r["bytes"] for r in rs)
            bad += sum(r["logged"]["bits_per_byte"][1] != r["bytes"] for r in rs)
            bad += not _close(-ll / nb / math.log(2), agg["bits_per_byte,none"])
            continue
        if any(key not in r["logged"] for r in rs):
            return False, -1
        if kind == "ll":
            acc = [float(r["greedy"]) for r in rs]
            bad += sum(float(r["logged"]["acc"]) != a for r, a in zip(rs, acc))
            bad += not _close(math.exp(-np.mean([r["ll"] for r in rs])), agg["perplexity,none"])
        else:
            acc = [float(np.argmax(r["ll"]) == r["gold"]) for r in rs]
            bad += sum("acc" in r["logged"] and r["logged"]["acc"] != a for r, a in zip(rs, acc))
            if "acc_norm" in rs[0]["logged"] and not rs[0]["multi"]:
                accn = [float(np.argmax(r["ll"] / r["n"]) == r["gold"]) for r in rs]
                bad += sum(r["logged"]["acc_norm"] != a for r, a in zip(rs, accn))
                bad += abs(np.mean(accn) - agg["acc_norm,none"]) > 1e-12
        bad += abs(np.mean(acc) - agg["acc,none"]) > 1e-12
    return bool(bad == 0), int(bad)


def check_t2(ra, rb):
    """Same examples, documents, prompts and continuations in both runs."""
    if set(ra) != set(rb):
        return False
    return all(ra[i]["doc_hash"] == rb[i]["doc_hash"] and ra[i]["prompt_hash"] == rb[i]["prompt_hash"]
               and ra[i]["args"] == rb[i]["args"] for i in ra)


def check_t5(recs, results):
    counts = {}
    for sub, _ in recs:
        counts[sub] = counts.get(sub, 0) + 1
    ns = results.get("n-samples", {})
    return all(sub not in ns or n == ns[sub]["effective"] == ns[sub]["original"] for sub, n in counts.items())


def load_pair(dir_a, dir_b, task, kind=None):
    ra, res_a = load_task(dir_a, task, kind)
    rb, res_b = load_task(dir_b, task, kind)
    (t1a, bad_a), (t1b, bad_b) = check_t1(ra, res_a, task, kind), check_t1(rb, res_b, task, kind)
    chk = {"T1": t1a and t1b, "T1_mismatches": bad_a + bad_b, "T2": bool(check_t2(ra, rb)),
           "T5": bool(check_t5(ra, res_a) and check_t5(rb, res_b))}
    if not (chk["T1"] and chk["T2"] and chk["T5"]):
        raise RuntimeError(f"integration test failed for {task}: {chk}")
    return ra, rb, chk


def metric_arrays(recs, task):
    kind, rule, _ = TASKS[task]
    ids = sorted(recs)
    ms = [example_metrics(recs[i], kind, rule) for i in ids]
    return {m: np.array([x[m] for x in ms]) for m in METRICS}, ids


# ---------------------------------------------------------------- statistics (unchanged from Revision 1)
def _boot(d, B, rng, chunk):
    """Bootstrap replicates of the mean and of the standardized mean (mean / sd) of d."""
    N, m, z = len(d), np.empty(B), np.empty(B)
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        x = d[rng.integers(0, N, size=(b, N))]
        mu, sd = x.mean(axis=1), x.std(axis=1, ddof=1)
        m[s:s + b], z[s:s + b] = mu, np.divide(mu, sd, out=np.zeros(b), where=sd > 0)
    return m, z


def _flip(d, B, rng, chunk):
    """Sign-flip (paired permutation) replicates of the mean and standardized mean under H0."""
    N, ss, m, z = len(d), float(d @ d), np.empty(B), np.empty(B)
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        signs = rng.integers(0, 2, size=(b, N), dtype=np.int8) * 2 - 1
        mu = (signs @ d) / N
        sd = np.sqrt(np.maximum(ss - N * mu ** 2, 0.0) / (N - 1))
        m[s:s + b], z[s:s + b] = mu, np.divide(mu, sd, out=np.zeros(b), where=sd > 0)
    return m, z


def _std(d):
    sd = d.std(ddof=1)
    return d.mean() / sd if sd > 0 else 0.0


def _p(null, obs):
    return (1 + np.sum(np.abs(null) >= abs(obs) * (1 - 1e-9))) / (len(null) + 1)


def task_stats(d, B=10_000, seed=0, chunk=250):
    rng = np.random.default_rng(seed)
    bm, bz = _boot(d, B, rng, chunk)
    fm, fz = _flip(d, B, rng, chunk)
    return {"N": len(d), "mean": d.mean(), "ci": np.percentile(bm, [2.5, 97.5]).tolist(), "p": _p(fm, d.mean()),
            "std_effect": _std(d), "std_ci": np.percentile(bz, [2.5, 97.5]).tolist(), "p_std": _p(fz, _std(d)),
            "frac_pos": float(np.mean(d > 0)), "frac_neg": float(np.mean(d < 0))}


def aggregate_stats(d_by_task, B=10_000, seed=0, chunk=250):
    """Equal task weights; stratified bootstrap and per-task sign flips, combined replicate by replicate."""
    rng, T = np.random.default_rng(seed), len(d_by_task)
    bm, bz, fm, fz = (np.zeros(B) for _ in range(4))
    for d in d_by_task:
        m, z = _boot(d, B, rng, chunk); bm += m / T; bz += z / T
        m, z = _flip(d, B, rng, chunk); fm += m / T; fz += z / T
    mean = float(np.mean([d.mean() for d in d_by_task]))
    std = float(np.mean([_std(d) for d in d_by_task]))
    return {"mean": mean, "ci": np.percentile(bm, [2.5, 97.5]).tolist(), "p": _p(fm, mean),
            "std_effect": std, "std_ci": np.percentile(bz, [2.5, 97.5]).tolist(), "p_std": _p(fz, std)}


# ---------------------------------------------------------------- additional statistics (Revision 2)
def contrast_stats(d1, d2, B=10_000, seed=0, chunk=250):
    """Difference of standardized effects, delta(d2) - delta(d1), for two independent tasks."""
    rng = np.random.default_rng(seed)
    _, z1 = _boot(d1, B, rng, chunk)
    _, z2 = _boot(d2, B, rng, chunk)
    return {"contrast": _std(d2) - _std(d1), "ci": np.percentile(z2 - z1, [2.5, 97.5]).tolist()}


def ratio_stats(s, c, B=10_000, seed=0, chunk=250):
    """Ratio sum(s)/sum(c) over clusters (documents) with cluster bootstrap CI and sign-flip p-value."""
    rng, N = np.random.default_rng(seed), len(s)
    est, bt, fl = s.sum() / c.sum(), np.empty(B), np.empty(B)
    for k in range(0, B, chunk):
        b = min(chunk, B - k)
        idx = rng.integers(0, N, size=(b, N))
        bt[k:k + b] = s[idx].sum(axis=1) / c[idx].sum(axis=1)
        signs = rng.integers(0, 2, size=(b, N), dtype=np.int8) * 2 - 1
        fl[k:k + b] = (signs @ s) / c.sum()
    return {"estimate": float(est), "ci": np.percentile(bt, [2.5, 97.5]).tolist(), "p": _p(fl, est), "N": N}


def ratio_contrast(s1, c1, s2, c2, B=10_000, seed=0, chunk=250):
    """sum(s2)/sum(c2) - sum(s1)/sum(c1) on the same clusters, cluster bootstrap CI."""
    rng, N = np.random.default_rng(seed), len(s1)
    bt = np.empty(B)
    for k in range(0, B, chunk):
        b = min(chunk, B - k)
        idx = rng.integers(0, N, size=(b, N))
        bt[k:k + b] = s2[idx].sum(1) / c2[idx].sum(1) - s1[idx].sum(1) / c1[idx].sum(1)
    return {"contrast": float(s2.sum() / c2.sum() - s1.sum() / c1.sum()),
            "ci": np.percentile(bt, [2.5, 97.5]).tolist()}


# ---------------------------------------------------------------- drivers
def admit(dir_a10, dir_a20, tasks=SUITE_Z):
    """Admission (Section 7): uses A only. ACC under each task's scoring rule, all subtasks pooled."""
    out = {"tasks": {}, "admitted": []}
    for t in tasks:
        r10, r20, chk = load_pair(dir_a10, dir_a20, t)
        a10, ids = metric_arrays(r10, t)
        a20, _ = metric_arrays(r20, t)
        N, acc10, acc20 = len(ids), a10["ACC"].mean(), a20["ACC"].mean()
        chance = 0.0 if TASKS[t][0] == "ll" else float(np.mean([1.0 / len(r20[i]["ll"]) for i in ids]))
        se = math.sqrt(acc20 * (1 - acc20) / N)              # binomial SE of the 20k accuracy
        d = a20["ACC"] - a10["ACC"]                            # same examples in both runs (T2)
        sd = d.std(ddof=1)
        above = (acc20 - chance) / se if se > 0 else 0.0
        gain_se = d.mean() / (sd / math.sqrt(N)) if sd > 0 else 0.0   # paired SE of the 10k->20k gain
        nonrandom, sensitive = bool(above >= 3), bool(gain_se >= 2)
        out["tasks"][t] = {"N": N, "chance": chance, "acc_10k": acc10, "acc_20k": acc20, "se_20k": se,
                           "above_chance_se": above, "gain": acc20 - acc10, "gain_se": gain_se,
                           "nonrandom": nonrandom, "sensitive": sensitive, "checks": chk}
        if nonrandom and sensitive:
            out["admitted"].append(t)
    return out


def analyze(dir_a, dir_b, tasks, B=10_000, seed=20261012, confirmatory=None):
    out = {"tasks": {}, "aggregate": {}, "checks": {}, "contrasts": {}}
    diffs = {}
    for k, t in enumerate(tasks):
        ra, rb, out["checks"][t] = load_pair(dir_a, dir_b, t)
        ma, _ = metric_arrays(ra, t)
        mb, _ = metric_arrays(rb, t)
        out["tasks"][t] = {}
        for m in METRICS:
            if np.isnan(ma[m]).any():
                continue
            d = mb[m] - ma[m]
            diffs[(t, m)] = d
            out["tasks"][t][m] = {"mean_A": ma[m].mean(), "mean_B": mb[m].mean(), **task_stats(d, B, seed + k)}
    sets = {"all": list(tasks)}
    if confirmatory is not None:
        sets["confirmatory"] = [t for t in tasks if t in confirmatory]
    for name, ts in sets.items():
        out["aggregate"][name] = {}
        for j, m in enumerate(METRICS):
            ds = [diffs[(t, m)] for t in ts if (t, m) in diffs]
            if ds:
                out["aggregate"][name][m] = {"tasks": [t for t in ts if (t, m) in diffs],
                                             **aggregate_stats(ds, B, seed + 1000 + j)}
    if all(t in tasks for t in PROBES):                     # hop contrasts (Section 8.3)
        for j, m in enumerate(("M1", "M2")):
            for t2 in PROBES[1:]:
                out["contrasts"][f"{m}:{t2}-{PROBES[0]}"] = contrast_stats(
                    diffs[(PROBES[0], m)], diffs[(t2, m)], B, seed + 2000 + 10 * j + PROBES.index(t2))
    if confirmatory is not None:                            # fixed-sequence test (Section 9)
        agg, c = out["aggregate"]["confirmatory"], sets["confirmatory"]
        s1 = len(c) >= 3 and "M1" in agg and agg["M1"]["p_std"] < 0.05
        h2 = s1 and "M2" in agg and len(agg["M2"]["tasks"]) >= 3 and agg["M2"]["p_std"] < 0.05
        out["confirmatory"] = {"set": c, "tested": len(c) >= 3, "H1_M1_rejected": bool(s1),
                               "H2_M2_rejected": bool(h2)}
    return out


def paloma(dir_a, dir_b, tasks=PALOMA, B=10_000, seed=20261012):
    """Bits-per-byte reduction BPB_A - BPB_B per source (byte-weighted), paired over documents."""
    out = {}
    for k, t in enumerate(tasks):
        ra, rb, chk = load_pair(dir_a, dir_b, t, kind="rolling")
        ids = sorted(ra)
        s = np.array([(rb[i]["ll"] - ra[i]["ll"]) / math.log(2) for i in ids])
        c = np.array([float(ra[i]["bytes"]) for i in ids])
        bpb = {arm: -sum(r[i]["ll"] for i in ids) / c.sum() / math.log(2) for arm, r in (("A", ra), ("B", rb))}
        out[t] = {"bpb_A": bpb["A"], "bpb_B": bpb["B"], "reduction": ratio_stats(s, c, B, seed + k), "checks": chk}
    return out


BUCKETS = [(1, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]


def position(loss_a, loss_b, buckets=BUCKETS, B=10_000, seed=20261012):
    """Loss reduction (A - B, nats/token) by token position; arrays [documents, positions], NaN = no token."""
    if loss_a.shape != loss_b.shape or not np.array_equal(np.isnan(loss_a), np.isnan(loss_b)):
        raise RuntimeError("loss arrays must cover the same documents and positions")
    diff = loss_a - loss_b
    out, sc = {}, {}
    for k, (lo, hi) in enumerate(buckets):
        seg = diff[:, lo:hi]
        s, c = np.nansum(seg, axis=1), np.sum(~np.isnan(seg), axis=1).astype(float)
        keep = c > 0
        sc[(lo, hi)] = (s, c)
        out[f"{lo}-{hi - 1}"] = ratio_stats(s[keep], c[keep], B, seed + k)
    (s1, c1), (s2, c2) = sc[buckets[0]], sc[buckets[-1]]
    keep = (c1 > 0) & (c2 > 0)
    out["last_minus_first"] = ratio_contrast(s1[keep], c1[keep], s2[keep], c2[keep], B, seed + 100)
    out["overall"] = ratio_stats(np.nansum(diff, 1), np.sum(~np.isnan(diff), 1).astype(float), B, seed + 200)
    return out


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("admit"); a.add_argument("--a10", required=True); a.add_argument("--a20", required=True)
    a.add_argument("--tasks", nargs="+", default=SUITE_Z); a.add_argument("--out", default="admission.json")
    p = sp.add_parser("pair"); p.add_argument("run_a"); p.add_argument("run_b")
    p.add_argument("--tasks", nargs="+", default=SUITE_Z); p.add_argument("--admission")
    p.add_argument("--B", type=int, default=10_000); p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--out", default="paired_result.json")
    q = sp.add_parser("paloma"); q.add_argument("run_a"); q.add_argument("run_b")
    q.add_argument("--tasks", nargs="+", default=PALOMA); q.add_argument("--out", default="paloma_result.json")
    r = sp.add_parser("position"); r.add_argument("loss_a"); r.add_argument("loss_b")
    r.add_argument("--out", default="position_result.json")
    args = ap.parse_args()
    if args.cmd == "admit":
        res = admit(args.a10, args.a20, args.tasks)
        for t, x in res["tasks"].items():
            print(f"{t:18s} N {x['N']:6d} acc10 {x['acc_10k']:.4f} acc20 {x['acc_20k']:.4f} chance {x['chance']:.3f} "
                  f"above {x['above_chance_se']:+6.1f} SE gain {x['gain_se']:+5.1f} paired SE -> "
                  f"{'ADMIT' if t in res['admitted'] else 'descriptive'}")
    elif args.cmd == "pair":
        conf = None
        if args.admission:
            with open(args.admission) as f:
                conf = json.load(f)["admitted"]
        res = analyze(args.run_a, args.run_b, args.tasks, args.B, args.seed, conf)
        for name, agg in res["aggregate"].items():
            for m, g in agg.items():
                print(f"{name:12s} {m:3s} D {g['mean']:+.6f} [{g['ci'][0]:+.6f}, {g['ci'][1]:+.6f}] p {g['p']:.4f} | "
                      f"delta {g['std_effect']:+.4f} [{g['std_ci'][0]:+.4f}, {g['std_ci'][1]:+.4f}] p {g['p_std']:.4f}")
        for t, ms in res["tasks"].items():
            for m, x in ms.items():
                print(f"{t:18s} {m:3s} A {x['mean_A']:+.5f} B {x['mean_B']:+.5f} d {x['mean']:+.6f} "
                      f"[{x['ci'][0]:+.6f}, {x['ci'][1]:+.6f}] p {x['p']:.4f} >0 {x['frac_pos']:.3f}")
        print(res.get("contrasts"), res.get("confirmatory"))
    elif args.cmd == "paloma":
        res = paloma(args.run_a, args.run_b, args.tasks)
        for t, x in res.items():
            g = x["reduction"]
            print(f"{t:40s} BPB A {x['bpb_A']:.5f} B {x['bpb_B']:.5f} reduction {g['estimate']:+.6f} "
                  f"[{g['ci'][0]:+.6f}, {g['ci'][1]:+.6f}] p {g['p']:.4f}")
    else:
        res = position(np.load(args.loss_a), np.load(args.loss_b))
        for k, x in res.items():
            v = x.get("estimate", x.get("contrast"))
            print(f"{k:16s} {v:+.6f} [{x['ci'][0]:+.6f}, {x['ci'][1]:+.6f}]" + (f" p {x['p']:.4f}" if "p" in x else ""))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
```

### A.3 T4 calibration

```python
import numpy as np
import paired_logprob_analysis as P

def t4(S=1000, B=2000, Ns=(1000, 1500, 2500), sig=(1.0, 0.1, 0.02), seed=7):
    rng = np.random.default_rng(seed)
    for eff in (0.0, 0.05):
        true_D, rej, cov = np.mean([eff * s for s in sig]), np.zeros(3), np.zeros(3)
        for k in range(S):
            d = [eff * s + s * rng.standard_t(5, N) / np.sqrt(5 / 3) for N, s in zip(Ns, sig)]
            g, t = P.aggregate_stats(d, B, 1000 + k), P.task_stats(d[0], B, 5000 + k)
            rej += [t["p"] < .05, g["p"] < .05, g["p_std"] < .05]
            cov += [t["ci"][0] <= eff * sig[0] <= t["ci"][1], g["ci"][0] <= true_D <= g["ci"][1],
                    g["std_ci"][0] <= eff <= g["std_ci"][1]]
        print(f"effect {eff}: rejection (task, D, delta) {rej / S}  coverage {cov / S}")
```

## Appendix B. Custom task configs

Put these six files in one directory and pass it with `--include_path` (Section 4.2). The bAbI configs share the template `_babi_cf_template_yaml`.

`_babi_cf_template_yaml`

```yaml
dataset_path: RMT-team/babilong-1k-samples
dataset_name: 0k
output_type: multiple_choice
doc_to_text: "{{input.strip()}}\nQuestion: {{question.strip()}}\nAnswer:"
doc_to_choice: ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
doc_to_target: "{{['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'].index(target.strip())}}"
num_fewshot: 2
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
```

`babi_cf_qa1.yaml`

```yaml
include: _babi_cf_template_yaml
task: babi_cf_qa1
test_split: qa1
fewshot_config:
  sampler: first_n
  samples:
    - input: "Daniel journeyed to the office. Sandra went to the garden."
      question: "Where is Daniel?"
      target: "office"
    - input: "John moved to the kitchen. Mary travelled to the bedroom. John went back to the hallway."
      question: "Where is John?"
      target: "hallway"
```

`babi_cf_qa2.yaml`

```yaml
include: _babi_cf_template_yaml
task: babi_cf_qa2
test_split: qa2
fewshot_config:
  sampler: first_n
  samples:
    - input: "Mary went to the kitchen. Mary got the apple there. Mary travelled to the office."
      question: "Where is the apple?"
      target: "office"
    - input: "Sandra went to the bathroom. Sandra grabbed the football there. Sandra journeyed to the garden. John moved to the hallway."
      question: "Where is the football?"
      target: "garden"
```

`babi_cf_qa3.yaml`

```yaml
include: _babi_cf_template_yaml
task: babi_cf_qa3
test_split: qa3
fewshot_config:
  sampler: first_n
  samples:
    - input: "John went to the bedroom. John got the milk there. John travelled to the kitchen. John went to the office."
      question: "Where was the milk before the office?"
      target: "kitchen"
    - input: "Mary moved to the garden. Mary picked up the apple there. Mary journeyed to the hallway. Mary dropped the apple. Daniel went to the bathroom."
      question: "Where was the apple before the hallway?"
      target: "garden"
```

`commonsense_qa_cf.yaml`

```yaml
task: commonsense_qa_cf
dataset_path: tau/commonsense_qa
output_type: multiple_choice
training_split: train
validation_split: validation
doc_to_text: "Question: {{question.strip()}}\nAnswer:"
doc_to_choice: "{{choices.text}}"
doc_to_target: "{{choices.label.index(answerKey)}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
```

`social_iqa_norm.yaml`

```yaml
task: social_iqa_norm
dataset_path: lighteval/siqa
dataset_name: null
output_type: multiple_choice
training_split: train
validation_split: validation
doc_to_text: "Q: {{context}} {{question}}\nA:"
target_delimiter: " "
doc_to_choice: "{{[answerA, answerB, answerC]}}"
doc_to_target: "{{ (label|int) - 1 }}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
```

## Sources

[1] Project results: `PROXY_RESULTS_SUMMARY.md` (snapshot 8 October 2026: 2.5k results, downstream protocol, reported accuracy keys and shot counts, fine-tuning seed spread), the seed-1042 few-shot and fine-tuning results at 10k and 20k, and the validation PPL at 10k and 15k–20k reported on 12 October 2026.

[2] Magnusson et al., *DataDecide: How to Predict Best Pretraining Data with Small Experiments*, ICML 2025 (arXiv:2504.11393).
- §2.5 defines CORRECT PROB, NORM CORRECT PROB, MARGIN and TOTAL PROB, normalized by characters unless stated otherwise.
- §3.3 finds these continuous metrics to be better or equivalent predictors of larger-scale decisions compared with accuracy. With them, MMLU, ARC, HellaSwag, MBPP and HumanEval become more than 80% predictable at the 1B target scale. GSM8K and Minerva do not benefit.

[3] Heineman et al., *Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation*, NeurIPS 2025 (arXiv:2508.13144). Switching from accuracy to a likelihood metric (perplexity or bits-per-byte) improves signal-to-noise, and so does averaging over a model's intermediate checkpoints. The paper calls ARC-Challenge too noisy: its accuracy ranges over 1.7 points across the final 30 checkpoints of 1B models.

[4] Koehn, *Statistical Significance Tests for Machine Translation Evaluation*, EMNLP 2004. Paired bootstrap resampling for comparing two systems on one test set.

[5] Dror, Baumer, Shlomov and Reichart, *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*, ACL 2018. How to choose significance tests in NLP, including bootstrap and permutation tests.

[6] EleutherAI lm-evaluation-harness 0.4.13, source files:
- `lm_eval/api/task.py`: `construct_requests` builds each continuation as the target delimiter followed by the choice; `process_results` divides by the choice length for `acc_norm` and takes WinoGrande's gold index from `doc_to_text`.
- `lm_eval/evaluator.py`: sample fields and hashes.
- `lm_eval/loggers/evaluation_tracker.py`: serialization of `arguments`, `filtered_resps` and `target`.
- `lm_eval/models/huggingface.py`: `prefix_token_id` is the configured prefix token, else BOS, else EOS; it is the context used for an empty prompt.
- Task configs: `belebele` and `commonsense_qa` score letter choices; `mmlu_continuation` is the cloze version of MMLU; `babilong` is a generation task with instructions; the Paloma README requires `logits_cache=False` with HF models.

[7] Westfall and Krishen, *Optimally weighted, fixed sequence and gatekeeper multiple testing procedures*, Journal of Statistical Planning and Inference 99 (2001). Testing hypotheses in a pre-specified fixed order controls the family-wise error rate without adjusting α.

[8] Penedo et al., *FineWeb2: One Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every Language*, arXiv:2506.20920 (2025), §3.3. Early-signal tasks are selected for monotonic improvement during training, low noise, non-random performance early in training, and ordering consistency.

[9] Gu et al., *OLMES: A Standard for Language Model Evaluations*, arXiv:2406.08446 (2024). Weaker base models need the cloze formulation, because their scores in the multiple-choice (letter) formulation hover around the random baseline. Recommended normalization: PMI for ARC-Challenge, CommonsenseQA and OpenBookQA; per-character for ARC-Easy, HellaSwag, PIQA, SocialIQA and MMLU; none for BoolQ and WinoGrande.

[10] Biran, Gottesman, Yang, Geva and Globerson, *Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries*, EMNLP 2024. In two-hop queries, the bridge entity is resolved in early layers and the second hop in later layers. Back-patching later-layer representations into earlier layers corrects 32–66% of previously incorrect cases.

[11] Magnusson et al., *Paloma: A Benchmark for Evaluating Language Model Fit*, NeurIPS 2024 Datasets and Benchmarks Track. The `allenai/paloma` dataset is gated under the AI2 ImpACT license.

[12] Warstadt et al., *BLiMP: The Benchmark of Linguistic Minimal Pairs for English*, TACL 2020. It has 67 paradigms of 1,000 minimal pairs each, scored by comparing sentence log-probabilities.

[13] Paperno et al., *The LAMBADA dataset: Word prediction requiring a broad discourse context*, ACL 2016.

[14] Weston et al., *Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks*, arXiv:1502.05698 (bAbI). Kuratov et al., *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack*, NeurIPS 2024 Datasets and Benchmarks Track (the `RMT-team/babilong-1k-samples` release, with config `0k` for the original stories).

[15] Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752. Its zero-shot evaluation reports LAMBADA (perplexity and accuracy), HellaSwag, PIQA, ARC-E, ARC-C and WinoGrande.
