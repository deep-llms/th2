# Local pretrained effectiveness screen — 2026-09-23

**Result: negative screen for a deep-source advantage.** All four deep-feedback
adapters improved held-out language-model loss over the frozen baseline, but
the matched shallow-attention adapter was slightly better for every pair.
All four 95% paired confidence intervals favor the shallow control. No pair
qualifies for the full PCC probe under section 11.2.1 of the research contract.

This is an exploratory result on a fixed local split of the user-specified
CulturaX source, not a reproduction of the unavailable finalized B200 split.
It provides no support for proceeding with this deep-feedback configuration
under the tested budget. It does not prove that every possible cross-depth
method fails, and it does not test a trained PCC student.

## Held-out results

NLL is token-weighted negative log likelihood in nats; lower is better.
Frozen Base: **NLL 3.02107552; perplexity 20.51334234**.

| Pair (s,d) | Deep NLL | Shallow NLL | Deep − shallow NLL | Paired 95% CI | Eligible |
|---|---:|---:|---:|---|---|
| (4,16) | 3.00114561 | 3.00096620 | +0.00017941 | [+0.00005072, +0.00031726] | No |
| (4,20) | 3.00111982 | 3.00096620 | +0.00015362 | [+0.00003160, +0.00028505] | No |
| (8,20) | 3.00449965 | 3.00400083 | +0.00049883 | [+0.00039265, +0.00061262] | No |
| (8,24) | 3.00483784 | 3.00400083 | +0.00083701 | [+0.00069825, +0.00097872] | No |

| Pair (s,d) | Deep perplexity | Shallow perplexity | Deep − Base NLL | Paired 95% CI |
|---|---:|---:|---:|---|
| (4,16) | 20.10856025 | 20.10495293 | −0.01992991 | [−0.02082883, −0.01894454] |
| (4,20) | 20.10804163 | 20.10495293 | −0.01995570 | [−0.02089040, −0.01897357] |
| (8,20) | 20.17611851 | 20.16605666 | −0.01657587 | [−0.01734636, −0.01580532] |
| (8,24) | 20.18294301 | 20.16605666 | −0.01623768 | [−0.01695715, −0.01547465] |

The deep adapters reduce perplexity by approximately 1.61–1.98% relative to
Base. That improvement alone is insufficient evidence for deep information:
the equally sized shallow branch achieves a larger gain. Shallow results for
the same s are identical because d does not enter that arm, and initialization,
data, optimization, and computation are matched.

Decision: `stop_negative_screen`; `selected_pair=null`. Per the existing rule,
no full PCC probe, target-permuted control, test evaluation, or pretraining was
launched. No new layer pairs or changed thresholds were tried after results.

## Fixed design and execution

- Backbone: pretrained `Qwen/Qwen3-0.6B-Base`, frozen, revision
  `ddc928429ed09d9ad603fd762053d0434c15e865`.
- Dataset: `nguyenhuuthuat09/CulturaX_sampled`, revision
  `b19d850278693d37113c197857cc6328fa5c6881`, file
  `raw/en/en_part_00015.parquet`. Source document rows [0,20000) train and
  [20000,30000) validation; these row ranges are disjoint.
- Legacy packing: tokenize without added special tokens/separators, concatenate
  each 1,000-document map batch, drop its remainder, then shuffle contexts with
  seed 20260922. Context length 2048. Every arm receives identical inputs.
- Each of eight adapters: 128 updates, 32,768 input tokens/update, 4,194,304
  total input tokens / 4,192,256 prediction targets. Microbatch 4 with four
  accumulation steps. Paired initialization seed 1701.
- Each evaluated arm: 2,000,000 input tokens / 1,999,023 prediction targets in
  977 contexts, including one partially padded context. Evaluation occurs once
  after the final update.
- Adapter: 1,051,649 trainable parameters. Existing optimizer, seven-update
  warmup, cosine schedule, bf16 computation and fp32 adapter parameters unchanged.
- Statistical rule: 1,000 paired context-bootstrap resamples, seed 20260922,
  shared resampled indices across all nine arms. Eligibility requires the upper
  95% CI of both deep−Base and deep−shallow to be below zero.
- Execution: dev host `transformer1`, four local A100-PCIE-40GB GPUs, one pair
  per GPU. Production `screen()` executed each pair; selection was deferred
  until the controller called the unchanged `select_pair()` over all results.
  No production training/statistics code was changed for this run.
- Wall time: 1,588.69 seconds (26.48 minutes), including worker setup and final
  decision; maximum allocated GPU tensor memory 6.23 GiB per worker. All workers
  exited successfully; read-only GPU inspection afterward found no compute jobs.
  B200, the sampler, and remote commands were untouched.

## Verification and limits

All four workers passed all 42 pretrained preflight checks. A separate artifact
audit verified exact 128-update sequences for all eight adapters, finite fp32
checkpoints, matched input/target counts, and reported NLL agreement with saved
per-context loss sums. Backbone fingerprints before training matched across
workers, and independently evaluated Base loss arrays were bit-identical across
all four GPUs. Source weights and parquet had previously been hash-verified
against pinned Hugging Face metadata.

This screen uses full-sequence scoring with `use_cache=False`. The earlier
native cached-vs-full bf16 logit comparison failed and remains recorded; it is
not the execution path measured here. The earlier one-update target-permutation
singleton failure also remains unresolved and was not exercised by this screen.
See [earlier diagnostics](PCC_DEV_DIAGNOSTICS_20260923.md).

## Evidence

Run root: `temp/pcc-promise-screen-20260923-a01/`.

- [Decision and exact metrics](../temp/pcc-promise-screen-20260923-a01/decision.json)
- [Successful completion](../temp/pcc-promise-screen-20260923-a01/complete.json)
- [Artifact audit](../temp/pcc-promise-screen-20260923-a01/artifact-audit.json)
- [Design fixed before training](../temp/pcc-promise-screen-20260923-a01/design.json)
- [Input counts and fingerprints](../temp/pcc-promise-screen-20260923-a01/data-check.json)
- [Local execution script](../temp/pcc-promise-screen-20260923-a01/run_screen.py)

Each `pair-s-d/` directory retains preflight/provenance, per-update logs, both
adapter checkpoints, per-context Base/deep/shallow loss files, correction/gate
diagnostics, and timings. These are local artifacts; no remote backup or push
was performed.
