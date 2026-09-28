# Deep-KV 30B budget: English train data shortfall — 2026-09-27

**Current data path:** training now uses train.py's cached Hugging Face maps
internally, with 160 training preprocessing workers and one validation worker.
The per-worker grouping boundaries can change remainders relative to this
single-worker estimate. The 28,600-update schedule remains, and training startup
checks the exact packed row count before any optimizer update. No separate
Deep-KV preparation command is required. See DEEP_KV_TRAINING.md.


**Status: recommended recipe fix applied; exact packed count remains unmeasured.**
The capacity fix uses 28,600 updates. On 2026-09-28 the user explicitly selected
1,430 warmup updates (5% of the full schedule), superseding the HF Trainer
migration's temporary 500-step warmup. Warmup has no effect on data capacity. The selected
four-arm run stops at 2,500 updates, preserving that full schedule. No resampling
or packing-policy change is needed. Training startup still checks the exact required
context count. The analysis below documents the superseded budget.

Commit `72901c7` set the Deep-KV recipe to
28,610 updates × 1,048,576 input tokens = 29,999,759,360 tokens per arm.
The completed English train split is expected to pack to about **302 fewer
2048-token contexts** than that budget requires. The former standalone
preparation command would therefore fail after tokenizing the whole split, before arm A starts.
This applies to cutoff queues too (e.g. `--stop-after 2000`), because
preparation always covers the full schedule so that later resumes keep the
same data order.

## Why the split has no margin

`prepare_data.py` (lines 305–345) adds shuffled English documents to train
until the running token count first reaches `target_tokens` = 30,000,000,000.
It then switches to eval until train + eval reach 30,010,000,000. Train
therefore holds 30B tokens plus less than one document of overshoot. The
counts use the same pinned tokenizer, without special tokens.

Deep-KV packing then changes the total in two ways (`deep_kv/data.py`,
`write_split`):

- It appends one `<|endoftext|>` per document (+36,595,514 tokens).
- It splits each 1,000-document batch into 2048-token contexts and drops that
  batch's remainder, about 1,024 tokens per batch on average.

These two effects almost cancel. The small net loss is enough to fall short.

## Estimate

Inputs are the sampler log line `[35/50] en_part_03060.parquet →
30,010,000,181 tokens so far`, identical on B200
(`temp/b200-sampling-complete-20260927-a02.log`) and in the local reproduction
(`temp/local-sampling-20260926-a01/sampling.log`), and the audited eval split
(`temp/deep-kv-real-eval-audit-20260927.json`: 11,822 documents, 10,011,667
tokens including one EOD each).

| Quantity | Value |
|---|---:|
| Eval tokens without EOD (10,011,667 − 11,822) | 9,999,845 |
| Train tokens without EOD (30,010,000,181 − 9,999,845) | 30,000,000,336 |
| Train documents / packing batches | 36,595,514 / 36,596 |
| Train tokens with one EOD per document | 30,036,595,850 |
| Expected dropped remainder (36,596 × 1,023.5) | 37,456,006 ± 113,098 |
| **Expected packed contexts** | **14,648,018 ± 55** |
| **Required contexts (28,610 × 512)** | **14,648,320** |
| Expected shortfall | 302 contexts ≈ 5.5 standard deviations |

The dropped remainder in each batch is its token total modulo 2048. That is
modelled as uniform on 0–2047 (mean 1,023.5, standard deviation ≈ 591), and
independent across batches. The eval split is consistent with this model: 12
batches dropped 11,283 tokens, 940 per batch, within about half a standard
deviation of the expected mean for a 12-batch sample.

This is an estimate, not a measured count. Only a full tokenization gives the
exact number. `write_split` detects a shortfall only after it has consumed
every batch. Based on the earlier local benchmark (~1.7M tokens/s), full
tokenization would take roughly 5 hours of CPU time before the error.

## Options

| Updates | Warmup | Tokens per arm | Required contexts | Expected margin |
|---:|---:|---:|---:|---:|
| 28,610 (current) | 1,431 | 29,999,759,360 | 14,648,320 | −302 (≈ −5.5 sd) |
| 28,608 | 1,431 | 29,997,662,208 | 14,647,296 | +722 (≈ 13 sd) |
| **28,600 (recommended)** | **1,430** | **29,989,273,600** | **14,643,200** | **+4,818 (≈ 87 sd)** |

**Recommended: 28,600 updates with 1,430 warmup updates** (5%, same rounding).
This shortens the budget by 0.03% and leaves a wide margin. It barely changes
a cutoff run. At update 2000 the learning rate is 2.99708e-4 with the current
recipe and 2.99707e-4 with the recommended one.

Other options, not recommended:

- **Keep 28,610.** Preparation is expected to fail after several hours.
- **Measure first, then set the budget.** This needs a separate full
  tokenization pass and makes the recipe depend on a data measurement.
- **Sample more English data.** This changes the completed sampler outputs
  and needs a new sampling run on B200.

## Scope of a fix

Any change must be committed before preparation or arm A. The recipe is part
of each prepared stream's manifest and each arm's run identity, and resume and
the report reject mismatches. The recommended fix changes `Recipe.updates` and
`Recipe.warmup` in `deep_kv/config.py`, the budget assertions in
`tests/test_deep_kv.py`, and the recipe tables in `DEEP_KV_TRAINING.md`,
`CURRENT_TASK.md` and `PROJECT_NOTES.md`. The mechanism, data order, packing
policy, eval split and cutoff behavior are unchanged. After a successful
preparation, `prepared/complete.json` records the exact context count.
