# Hashed (Product Code H2048) — Chinese-failure diagnostics

Status: 2026-09-02. Implemented; production checkpoint checks have not yet
been run. Checks 1/2/6 use the existing `eval/ppl_bytoken.py` and
`eval/ppl_bins.py` paths. Checks 3--5 are implemented in
`eval/hashed_zh_diagnostics.py`. The focused unit suite and a tiny real
Product-Code checkpoint train/load/eval fixture pass on CPU; these are code
verification, not Hashed checkpoint-10k results.
Applies to the completed `product_code_hashed_h2048` checkpoint-10k run
(results: `docs/PROJECT_NOTES.md` § "2026-09-02 — Hashed and Residual
Subspace Experts"; method: `docs/product_code_design.md`).

## 1. The question

Hashed nearly matches the best compressed control on five languages and
fails on one:

| | en | ar | de | ru | vi | zh | mean |
|---|---|---|---|---|---|---|---|
| dense | 22.37 | 17.45 | 20.36 | 13.78 | 15.24 | 58.53 | 24.62 |
| GroupReduce control | 25.04 | 19.81 | 24.06 | 15.91 | 17.02 | 86.17 | 31.33 |
| Hashed H2048 | 26.07 | 19.76 | 24.63 | 15.93 | 17.16 | **112.67** | 36.04 |

Two hypotheses explain a Chinese-only failure and call for different fixes:

- **H-coverage.** Chinese needs more private (head) rows than any other
  language; too much of its probability mass is represented by codebook
  sums. Fix: larger or language-aware head.
- **H-interference.** Codebook rows shared by ~36 unrelated tokens cannot
  represent tokens that each carry real mass; Chinese is the only language
  whose mass mostly lives in the tail, so it alone pays. Fix: codes that
  group similar tokens (teacher PQ) or language-partitioned codebooks.

A third possibility, **H-global** (the Chinese head tokens are also worse,
i.e. the failure is not in the codebooks at all), must be excluded first.

## 2. What is already known without the checkpoint (computed 2026-09-02)

From `resources/token_freq_sample10.npz` (per-language counts, keys
`counts_<lang>`) and the head/tail partition actually used by the run
(`compositional.product_code.head_tail_partition` on
`resources/token_importance_langbalanced.npz`, head size 2,048; codes from
`hashed_codes(tail_ids, 4, 4096, seed=0)`):

| language | token mass in the head | effective vocabulary `exp(H)` | rows for 80% of mass | head rows whose dominant language this is |
|---|---|---|---|---|
| en | 54% | 2,357 | 4,462 | 198 |
| ar | 73% | 928 | — | 391 |
| de | 57% | 2,024 | — | 336 |
| ru | 68% | 1,134 | — | 395 |
| vi | 71% | 1,217 | — | 484 |
| zh | **39%** | **3,568** | **4,476** | 244 |

- 61% of Chinese mass goes through codebook rows, versus 27–46% for every
  other language. The control puts the same 61% of Chinese mass in its
  rank-64 and rank-192 blocks (32% + 29%), which is why the control is also
  the worst on Chinese; Hashed is worse still for that mass.
- Each Chinese tail token shares each of its 4 rows with 35.6 other tokens,
  5.4 of them Chinese; mass-weighted, 39% of the mass in a Chinese token's
  buckets is Chinese although Chinese is one sixth of the balanced eval.
- Training loss at the end was *lower* for Hashed (3.313) than RSE (3.370):
  the model fits the 85.7%-English training stream well; the failure is on
  the held-out Chinese distribution.

These facts make both hypotheses plausible and do not separate them. The
checks below do.

## 3. Checks, in order

All by-token outputs are `eval_ppl_bytoken.npz` files (per language:
`<lang>_nll` float64 sums and `<lang>_cnt` int64 counts per token id) written
by `eval/ppl_bytoken.py`; binning is `eval/ppl_bins.py`. The dense and
GroupReduce-control by-token files already exist from the 2026-08-31
diagnostic and are reused as references.

### Check 1 — Is the failure confined to tail (codebook) tokens? (excludes H-global)

By-token PPL on the Hashed checkpoint, all six languages:

```
python eval/ppl_bytoken.py --checkpoint <hashed ckpt-10000> --eval-dir <eval-dir> \
    --tokenizer-name <tokenizer> --bf16 --output-dir <out>/hashed
```

Then bin with the **language-balanced** ordering and the run's actual
head/tail split (populations `2048,149888`), Chinese only and all languages:

```
python eval/ppl_bins.py --counts resources/token_importance_langbalanced.npz \
    --populations 2048,149888 --reference dense \
    --run dense=<dense bytoken.npz> --run control=<control bytoken.npz> \
    --run hashed=<out>/hashed/eval_ppl_bytoken.npz --langs zh --output <out>/bins_zh_headtail.json
```

Record, for zh and for the non-zh mean: PPL of head tokens and of tail
tokens for dense / control / Hashed, and the share of the Chinese gap
(Hashed vs dense) contributed by each bin.

`ppl_bins.py` writes these values in `per_language_ppl`,
`mean_per_language_ppl`, `per_language_gap_vs_reference`, and
`mean_per_language_gap_vs_reference`. Run once with `--langs zh` for the
Chinese table and once with `--langs en ar de ru vi` for the equal-language
non-Chinese summary; do not use the pooled token-weighted row as the
non-Chinese language mean.

Reading:
- Hashed head-bin PPL on zh ≈ control's head-bin PPL → the failure is in the
  tail; continue.
- Hashed head-bin PPL on zh clearly worse than the control's → H-global:
  the shared training (bias, gates, or the head rows being starved by the
  much larger tail gradient) hurts Chinese even where representation is
  private. Then Checks 5 and 6 matter most and the codebook checks are
  secondary.

### Check 2 — Within the tail, does the loss grow with importance rank? (H-coverage vs H-interference)

Same by-token files, finer level sets under the language-balanced ordering
(the default populations `2048,6144,24576,119168` split the tail into three
importance tiers), plus mass bins:

```
python eval/ppl_bins.py --counts resources/token_importance_langbalanced.npz \
    --populations 2048,6144,24576,119168 --mass-thresholds 0.5,0.8,0.9,0.99 \
    --reference control --run control=... --run hashed=... --langs zh --output <out>/bins_zh_tiers.json
```

Record Hashed-vs-control PPL ratio per tail tier for zh, and the same for
the other five languages.

Reading:
- Ratio is modest for the first tail tier (tokens ranked 2,049–8,192) and
  grows sharply toward the low-importance tiers → capacity per token:
  H-coverage. A larger head would have absorbed the first tiers.
- Ratio is roughly flat across tiers, and specific to zh (other languages'
  tail tiers are near 1.0) → the representation itself fails for Chinese
  tokens regardless of their rank: H-interference.
- Also compare the same tiers under the **raw-frequency** ordering
  (`--counts resources/token_freq_sample10.npz`) so the numbers are
  comparable with the control's 2026-08-31 table (+4.1/+10.4/+30.4/+108%).

### Check 3 — Does the model confuse Chinese targets with their bucket-mates? (direct H-interference test)

Implemented by `eval/hashed_zh_diagnostics.py`, run on the checkpoint with the
Chinese and German eval splits. It uses the same window boundaries, stride,
label shift, and `-100` context masking as `ppl_bytoken.py`, and fails if its
manual loss differs from the model loss by more than `1e-3`:

```
python eval/hashed_zh_diagnostics.py \
    --checkpoint <hashed ckpt-10000> \
    --eval-dir <eval-dir> --tokenizer-name <tokenizer> --langs zh de --bf16 \
    --language-counts resources/token_freq_sample10.npz \
    --importance resources/token_importance_langbalanced.npz \
    --bytoken <out>/hashed/eval_ppl_bytoken.npz \
    --output <out>/hashed_zh_checks_3_4_5.json
```

Omit `--eval-dir`, `--tokenizer-name`, and `--langs` to run only the cheaper
state-dict Checks 4/5. `--max-positions` exists only for smoke testing and
defaults to `0` (all positions) for the production diagnostic. The random
baseline is deterministic per target and excludes the target plus all of its
bucket-mates.

The prediction path performs the following:

1. Load the model with `compositional.loading.load_compositional_model`;
   reject any checkpoint that is not the hashed assignment with the exact
   Product-Code tied head pointing at this same embedding module, and
   regenerate the codes from the declared hash seed to require an exact match;
   read `embed.head_ids`, `embed.tail_ids`, `embed.codes` (shape
   `[tail_size, 4]`).
2. For every evaluation position whose target is a Chinese **tail** token,
   compute the softmax over the full vocabulary and accumulate:
   - `p_target`;
   - `p_mates`: total probability on the other tail tokens sharing ≥1 code
     with the target (the union of its four buckets, ≈140 tokens);
   - `p_random`: total probability on a fixed random set of tail tokens of
     the same size (seeded once per target);
   - the rank of the target and whether the arg-max is a bucket-mate.
3. Report the means of `p_mates / p_random` and of `p_mates / p_target`,
   the fraction of positions whose arg-max is a bucket-mate, and the same
   statistics for a non-Chinese language (de) as a control.

Reading: `p_mates / p_random` ≫ 1 for zh (and ≈ 1–2 for de) means the
mixture is leaking mass to tokens that share rows — H-interference
confirmed at the prediction level. ≈ 1 means the shared rows are not
where the mass goes and the failure is capacity.

### Check 4 — Have bucket-mates collapsed to near-identical rows? (H-interference at the parameter level; state dict only)

Same script, no eval data needed:

1. Materialize the effective table (`embed.materialize()`, fp32) for all
   tail tokens.
2. Cosine similarity between (a) pairs of tail tokens sharing ≥1 code,
   (b) pairs sharing 2+ codes, (c) random tail pairs — each split into
   zh–zh, zh–other, other–other pairs by dominant language (arg-max of the
   per-language normalized share from `token_freq_sample10.npz`). Random
   controls are language-category matched and explicitly share no code.
3. Also the norm distribution of tail rows, zh vs other, and of head rows.

Reading: bucket-mate similarity far above the random-pair baseline,
especially for 2+-code pairs and for zh–zh pairs, means the four scalar
gates do not separate identities; the model cannot tell those tokens apart
in any context. Similar-to-baseline means identities are fine and the
problem is elsewhere.

### Check 5 — What did the gates learn? (state dict only)

From `embed.gate_offsets` (`[tail_size, 4]`, effective gate = 1 + offset):

- Distribution (mean, std, 1st/99th percentile) of gates for zh-dominant vs
  other tail tokens, and vs importance rank.
- Fraction of tokens with any gate below 0.1 (row effectively switched off)
  or above 3 (row dominating), zh vs other.
- Correlation of gate magnitude with the token's per-token PPL from Check 1.

Reading: extreme or switched-off gates concentrated on Chinese tokens show
the model fighting the shared rows (supports H-interference); gates near 1
everywhere show the gate parameterization is not being used, which by
itself suggests that four scalars are too weak a per-token identity.

### Check 6 — When does Chinese diverge from the control during training?

Run `eval/ppl_bytoken.py --langs zh` (or `eval/ppl.py` on the zh split) on
the saved intermediate checkpoints (every 250 steps; use 1000, 2500, 5000,
7500, 10000) of Hashed and of the GroupReduce control, and plot zh PPL vs
step for both, alongside en.

Reading:
- Gap present from the earliest checkpoint and roughly constant in ratio →
  capacity/representation (either hypothesis, resolved by Checks 2–4).
- Gap that widens late while en keeps improving → rows specializing toward
  the English-dominated training stream at the expense of the shared
  Chinese tail (interference growing with training); in that case a second
  seed would not help and language partitioning of the codebooks is the
  targeted fix.

## 4. Decision table

| outcome | conclusion | targeted follow-up (if any) |
|---|---|---|
| Check 1: zh head bin also worse than control | H-global | Study Check 5/6 first; no codebook change is justified until the head-row deficit is explained |
| Check 2: loss grows with tail rank; Check 3 ratio ≈ 1 | H-coverage | Head sized by per-language mass coverage (e.g. rows until each language reaches 80% mass; ≈ 4.5k for zh alone) — the 8,192-row fallback config in `product_code_design.md` §2.2 (19,450,368 params) is the ready-made test |
| Check 2 flat; Check 3 ratio ≫ 1; Check 4 mates collapsed | H-interference | Language-partitioned codebooks (Chinese tail tokens share rows only with Chinese tokens; zero parameter change) or teacher-PQ codes (`scripts/make_pq_codes.py`, groups similar tokens deliberately) |
| Check 6 gap widens late | interference driven by training mix | Same as above; also report per-language training loss in any rerun |

Whatever the outcome, the run's measured 1.5× dense wall-clock (8.57 h vs
5.4–5.6 h) is unchanged by any of these fixes: it comes from the exact
tied output materializing the full table (`product_code_design.md` §2.6).
A follow-up run must state up front that it accepts that cost or pairs the
fix with a cheaper head.

## 5. Deliverables to record in `docs/PROJECT_NOTES.md`

1. The Check-1 head/tail table for zh and the non-zh mean (dense, control,
   Hashed) with gap shares.
2. The Check-2 tier ratios, under both orderings.
3. Check-3 statistics for zh and de; Check-4 similarity table; Check-5 gate
   summary.
4. The Check-6 curve values at the five steps.
5. One paragraph naming which hypothesis survived and which row of the
   decision table applies.
