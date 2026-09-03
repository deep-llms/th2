# Hashed v2 — Coverage-quota head for the Product-Code interface

Status: 2026-09-02. **Implemented and locally validated; not launched.**
Successor to the completed `product_code_hashed_h2048` run (mean PPL 36.04,
Chinese 112.67; `docs/PROJECT_NOTES.md` § 2026-09-02). The v1 checkpoint was
lost in a machine restart, so the Chinese diagnostics
(`docs/hashed_zh_diagnostics.md`) will run on the v2 checkpoint instead.
Accuracy only: the speed items (per-step materialization, fused
product-code logit kernel) are deliberately out of scope and listed in §9.

---

## 1. Idea as first proposed, and what the judging changed

Proposed: (A) select the private head rows so that every language reaches
the same fraction of its token mass in private rows ("coverage quota"),
enlarging the head to 6,144–8,192 rows; (B) hash each tail token only into
buckets reserved for its dominant language ("language-partitioned codes").

Judged outcome:

| item | verdict | reason |
|---|---|---|
| A. coverage-quota head | **kept, modified** | Directly addresses the one measured structural asymmetry (§2). Modified to a *floor* quota so no language's coverage drops below what v1 gave it (§3.2). |
| B. language-partitioned codes | **dropped from this run** | Direction of effect is not clear (§1.1); confounds attribution; needs a fuzzy dominant-language label for shared subwords, digits, punctuation. Decided by Check 3 of the diagnostics on the v2 checkpoint, not by assumption. |
| loss-weighted head selection (gap per token from the dense/control by-token files) | **rejected** | Those files are computed on the *evaluation* split; selecting head tokens from them is test-set leakage. Training-side counts only. |
| larger codebooks instead of a larger head | rejected | v1 already had B=4,096 (37 tail tokens per row) and failed on the language whose mass is in the tail; the head is the lever the numbers point at. |
| teacher-PQ codes | not available | requires a dense checkpoint on the node; lost in the same restart. |

### 1.1 Why partitioning is not obviously positive

A codebook row is an additive component shared by ~37–48 tokens. Sharing
costs most when bucket-mates *compete in the same softmax*, i.e. appear in
the same contexts. Random codes make a Chinese token's bucket-mates mostly
non-Chinese tokens that carry near-zero probability in Chinese contexts;
partitioning would replace them with Chinese tokens that do compete. So
partitioning could plausibly *increase* the interference that matters,
while reducing the cross-language kind that may be nearly free. That is an
empirical question the bucket-mate confusion check answers; it should not
be bundled into a run whose purpose is to test coverage.

### 1.2 What survives: the coverage asymmetry is measured, not hypothesized

From training counts only (`resources/token_freq_sample10.npz`):

| language | mass in v1's 2,048-row head | effective vocabulary | v1 head rows |
|---|---|---|---|
| en | 54% | 2,357 | 198 |
| ar | 73% | 928 | 391 |
| de | 57% | 2,024 | 336 |
| ru | 68% | 1,134 | 395 |
| vi | 71% | 1,217 | 484 |
| zh | **39%** | **3,568** | 244 |

61% of Chinese mass, and 43–46% of English and German mass, is
represented by codebook sums. Those are the three languages where v1 trails
the control (zh by 26.5, en by 1.03, de by 0.57 PPL); the three languages
with ≥ 68% head coverage are at or better than the control. That is one
consistent story; the head is where it points.

---

## 2. Method (unchanged module; new head-selection ordering)

`ProductCodeEmbed` selects the head as the top-`H` tokens of an importance
vector under the stable `(-importance, token_id)` rule
(`compositional.product_code.head_tail_partition`); hashed codes depend only
on tail token ids. Therefore the whole change is **a new importance file
plus three launch flags**, with no change to any module, trainer, loader,
or test. Everything verified for v1 (`docs/product_code_design.md` §13–14)
carries over.

### 2.1 Floor-quota ordering

Inputs: per-language normalized shares `s_l(w) = counts_l(w) / Σ_w counts_l(w)`
for the six languages (training sample counts), and v1's head set `𝓗₀`
(top-2,048 of `token_importance_langbalanced.npz`).

Ordering:
1. `𝓗₀` first, in v1's language-balanced importance order (so the first
   2,048 positions reproduce v1's head exactly).
2. Then greedily: repeat — find the language with the lowest current
   coverage `c_l = Σ_{w selected} s_l(w)`; append its highest-share token not
   yet selected (stable by token id); update all six coverages.
3. Continue the greedy pass until every token with nonzero mass in any
   language is placed; tokens with zero mass everywhere last, by id.

Emit `importance(w) = V − position(w)` (float64, key `counts`), which is
finite, nonnegative and strictly decreasing in position, so
`head_tail_partition(importance, H)` returns exactly the first `H`
positions for any `H`. The same file serves `eval/ppl_bins.py --counts`
for level-set bins `2048, H−2048, V−H` = v1-head / added rows / tail.

Deterministic; no RNG. The provenance JSON records the source SHA256, the
per-language coverage at `H ∈ {2048, 4096, 6144, 8192}`, and the head
composition by dominant language.

### 2.2 Reference configuration

Budget rule as v1: total ≤ the LR128 envelope, in practice ≤ v1's
19,474,944.

| config | head `H` | buckets `B` (H=4 codebooks) | params | coverage per language | tail tokens per row |
|---|---|---|---|---|---|
| v1 (measured) | 2,048 | 4,096 | 19,474,944 | 39–73% | 36.6 |
| v2-4096 | 4,096 | 3,584 | 19,466,752 | 64–73% | 41.2 |
| **v2-6144 (primary)** | **6,144** | **3,072** | **19,458,560** | **71–73%** | 47.5 |
| v2-8192 (fallback) | 8,192 | 2,560 | 19,450,368 | 75.5% all | 56.1 |

Primary head composition by dominant language at H=6,144:
en 1,498 · ar 393 · de 873 · ru 469 · vi 491 · zh 2,420. The 4,096 added
rows go to zh (2,176), en (1,300), de (537), ru (74); ar and vi gain
almost nothing because they are already above the common level, and lose
nothing because of the floor.

Why 6,144 and not 8,192: the head is paid for by codebook rows, and
tail-tokens-per-row rises from 37 to 48 (6,144) or 56 (8,192). If
interference matters at all, the 8,192 config spends more of it; 6,144 is
the balance point, with 8,192 as the pre-registered fallback if v2-6144
improves Chinese but not enough (§6).

---

## 3. Expected outcome (written before the run)

Chinese NLL excess over dense in v1 is 0.654 nats/token. If all of it sits
in the 61% of mass carried by codebooks and the per-mass excess stays
constant, cutting the codebook share to 29% (H=6,144) gives zh ≈ 80; to
24.5% (H=8,192) gives zh ≈ 76. That is optimistic: the rows moved into the
head are the *easiest* tail tokens, and the remaining tail gets denser.
Realistic: **zh 80–95** (control 86.17).

English and German gain coverage (54 → 71%, 57 → 71%); expected
−0.3 to −1.0 PPL each. Arabic, Russian, Vietnamese: unchanged coverage,
slightly denser buckets; expected within ±0.3 of v1.

| | v1 (measured) | v2-6144 expected |
|---|---|---|
| zh | 112.67 | 80–95 |
| non-zh mean | 20.71 | 20.2–20.7 |
| **six-language mean** | 36.04 | **30.4–32.5** |

So the honest expectation is that v2 lands **around the GroupReduce
control (31.33), not at the 29.5 pass bar.** Reaching 29.5 with the other
five languages unchanged would need zh ≤ 73, i.e. better than the control
on the language the control is worst at. This is stated so the result is
judged against it.

---

## 4. Pre-registered criteria

Primary bar unchanged from `docs/product_code_design.md` §7: **mean PPL
≤ 29.5** (strong ≤ 28.5), measured with the standard six-split eval.

Informative criteria, fixed now, to interpret a miss:

- **C1 (coverage was the cause):** zh ≤ 90 *and* no other language worse
  than v1 by more than 0.3 PPL. Then the Chinese failure is explained by
  head coverage, and what remains is the ordinary compressed-tail gap.
- **C2 (coverage was not the cause):** zh ≥ 100. The head-coverage
  hypothesis is refuted; the bucket-mate confusion and row-similarity
  checks (`hashed_zh_diagnostics.md` Checks 3–5, on this checkpoint) decide
  between interference and something global.
- **C3 (side-effects):** ar/ru/vi each within ±0.3 of v1. A regression
  here means denser buckets hurt, and the fallback config (§2.2, denser
  still) is *not* the next step.

Decision table:

| outcome | next |
|---|---|
| mean ≤ 29.5 | pass: second seed, then the speed work (§9) becomes worth doing |
| C1 but mean > 29.5 | family ≈ control at dense head compute: not a result by itself. Continue only if the fused-kernel path (§9) is accepted as the efficiency story; otherwise stop the family |
| C1 and 29.5 < mean ≤ 30.5 and C3 holds | one fallback run v2-8192 is justified |
| C2 | no further coverage runs; interference follow-up only if Check 3 confirms it |
| C3 fails | stop increasing the head; bucket density is a real cost |

---

## 5. Diagnostics on the v2 checkpoint

Run the full battery in `docs/hashed_zh_diagnostics.md` on the v2
checkpoint-10k, with the level-set bins `2048,4096,141696` under the v2
importance file (v1 head / added rows / tail) *and* under the language-
balanced file with `2048,149888` (comparable with the v1 analysis). Keep
intermediate checkpoints (every 250 steps) until Check 6 has been run.

---

## 6. Runs

- **Run v2-6144** (`product_code_quota_h6144`): the v1 launch script
  (`scripts/train_product_code_tied.sh`) with
  `--product_code_head_size 6144 --product_code_num_buckets 3072
  --product_code_importance_path resources/token_importance_quota.npz`;
  everything else identical (seed 42, 16 × 4, bf16, 10k steps, same
  handoff: six-split eval → by-token PPL → 26 zero-shot tasks → finetune
  battery).
- **Run v2-8192** (`product_code_quota_h8192`): only under the §4 decision
  table.

Order relative to BT-MoS: BT-MoS's 100-step gate is short and should go
first on the node; the two 10k runs then queue in whichever order the user
chooses. They do not share code paths.

---

## 7. Implementation steps (small; no module change)

1. `scripts/make_token_importance_quota.py` — builds the §2.1 ordering from
   `resources/token_freq_sample10.npz` (keys `counts_<lang>`) and
   `resources/token_importance_langbalanced.npz` (for `𝓗₀`); writes
   `resources/token_importance_quota.npz` (`counts`, float64, shape `(V,)`)
   and `.json` provenance (source SHA256s, coverage table at the four head
   sizes, composition, first/last placed ids). Assert: the first 2,048
   positions equal v1's head set; coverages are non-decreasing in `H`; every
   id appears exactly once.
2. `scripts/train_product_code_quota_tied.sh` — copy of the v1 script with
   the three flags and a new output dir; `test -s` on the new `.npz`.
3. `run_experiments.py` — append `product_code_quota_h6144` and
   `product_code_quota_h8192` **last**; `required_input_files` = the new
   `.npz`.
4. Tests (`compositional/test_product_code.py` additions): the importance
   file loads through `load_frequency_counts` with pseudocount 0; the
   production build at H=6,144/B=3,072 counts 19,458,560 parameters
   through the real `build_arm` path; `head_tail_partition` on the file at
   H=2,048 reproduces the v1 head ids.
5. `scripts/smoke_product_code_e2e.py` already covers train/resume/eval on
   this arm; rerun it with the new flags (head 6,144 scaled to the fixture
   is unnecessary — the fixture tests the code path, the count test covers
   the config).
6. Commit the `.npz` + `.json` with the code (both required by the launcher).

### 7.1 Implementation record (2026-09-02)

Implemented without changing `ProductCodeEmbed`, the tied head, trainer, or
checkpoint loader:

- `scripts/make_token_importance_quota.py` implements the deterministic
  floor-quota pass, validates the permutation/v1-prefix/zero-mass-tail and
  monotonic-coverage invariants, and emits both required artifacts.
- `resources/token_importance_quota.npz` contains `counts` as float64 with
  shape `(151936,)`; SHA256
  `f43d19925f5add96c56913eccf57f3989d6cd52e69da761d879e22f901010ea5`.
  Byte-for-byte regeneration with the current source artifacts was verified.
- `resources/token_importance_quota.json` records both input hashes, the
  output hash, coverage and dominant-language composition at all four
  registered head sizes, the v1-head id hash, and boundary token ids.
- `scripts/train_product_code_quota_tied.sh` defaults to the primary
  H=6,144/B=3,072 configuration. Setting
  `PRODUCT_CODE_QUOTA_HEAD_SIZE=8192` selects the pre-registered
  H=8,192/B=2,560 fallback; other values are rejected before launch.
- `run_experiments.py` registers `product_code_quota_h6144` and
  `product_code_quota_h8192` as the final two entries and requires the quota
  artifact before any GPU cleanup or launch.

Validation evidence: the focused generator and Product-Code suites passed 23
tests, with one CUDA-only test skipped on the test runner (21 Product-Code
tests plus 2 quota-generator tests). The production
`build_arm` path constructed H=6,144/B=3,072 and counted exactly 19,458,560
trainable parameters. The H=2,048 quota prefix exactly reproduced the ordered
v1 head. The existing end-to-end smoke remains the correct train/resume/eval
code-path test because this version changes only the input ordering and
production dimensions, which the artifact and production-build tests cover.
Mocked launcher checks also verified that the two runner entries emit exactly
H=6,144/B=3,072 and H=8,192/B=2,560 with distinct matching output directories;
both entries set the head size explicitly so an inherited shell variable
cannot change the registered experiment.
A production-shape H=6,144/B=3,072 bf16 CUDA smoke on an A100 passed tied-head
forward/backward with finite gradients for every embedding parameter and
confirmed the same 19,458,560-parameter count; the GPU was free afterward.

---

## 8. Risks

1. **Denser buckets.** 37 → 48 tail tokens per row. If v1's non-Chinese
   strength depended on sparse sharing, ar/ru/vi regress (C3 catches it).
2. **The moved tokens are the easy ones.** The §3 arithmetic assumes
   constant excess per unit of tail mass; the real remaining tail is
   harder per token. That is why the expectation is a range, not 80.
3. **Head-row starvation.** 6,144 private rows now receive gradient from
   fewer training positions each (they are language-balanced, and training
   is 85.7% English); Chinese head rows see Chinese data only 1.4% of the
   time. Cannot be fixed at the interface; it is the same for the control.
4. **Attribution.** Only one change is made (head selection and size,
   with the codebook shrink it forces), so the result attributes cleanly to
   coverage — this is the reason partitioning was dropped.
5. **Compute.** Head compute is still dense-level (materialize + full GEMM),
   and v1's measured 1.5× dense wall-clock stands. Accepted for this run by
   the user's decision to fix accuracy first; §9 is the plan for it.

---

## 9. Out of scope here (recorded so it is not lost)

- **Speed, level 1 (code only, ceiling = dense speed):** profile 20 steps;
  materialize the table once per optimizer step (a detached leaf whose
  gradient accumulates across the four microbatches, then one backward
  through `materialize`), and replace the atomic `index_add_` codebook
  scatter with a sort-based segment reduction.
- **Speed, level 2 (fused kernel, below dense):** never materialize; per
  position compute the four `B`-wide bucket score slabs (`4·B·d` MACs,
  12.6M at B=3,072) plus the head GEMM (6.3M), and assemble each logit as a
  gated sum of four gathered slab entries in a Triton kernel fused with the
  cross-entropy. ≈ 0.12× dense head MACs. Forward + backward kernels with
  bit-level agreement tests against `TiedProductCodeHead`. Only worth
  building if v2 passes.
- **Language-partitioned or teacher-PQ codes:** after Check 3.
- **Phase 2 learned code assignment:** unchanged from
  `product_code_design.md` §10.
