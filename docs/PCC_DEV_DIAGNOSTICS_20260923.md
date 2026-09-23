# Local pretrained diagnostics — 2026-09-23

Diagnostics ran on dev host `transformer1`, physical GPU 0 (NVIDIA A100-PCIE-40GB),
using `train_env`. Read-only checks found four idle A100s; only GPU 0 was used.
It was idle again after completion. No B200 access or sampler changes occurred.

## Inputs

- Model: `Qwen/Qwen3-0.6B-Base`, pinned revision
  `ddc928429ed09d9ad603fd762053d0434c15e865`.
- User-selected dataset: [nguyenhuuthuat09/CulturaX_sampled](https://huggingface.co/datasets/nguyenhuuthuat09/CulturaX_sampled),
  revision `b19d850278693d37113c197857cc6328fa5c6881`.
- Downloaded English shard: `raw/en/en_part_00015.parquet`.
- Model weights (1,192,135,096 bytes) and parquet (2,496,438,120 bytes) were
  SHA256-verified against the pinned Hub metadata. Downloaded files and exact
  local paths are retained under `temp/dev-diagnostics-20260923-a01/`.

The repository contains raw parquet shards, not the final train/validation sets
being sampled on B200. This diagnostic tokenized the first 128 documents and
used the first 16 complete contexts, with no special tokens or context shuffle.
It did not reproduce the fixed scientific training stream or select a dev/test split.

## Results

| Check | Result |
|---|---|
| Tiny CPU correctness preflight | All 42 checks passed |
| Exact pretrained-model correctness preflight on A100 | All 42 checks passed |
| Privileged teacher optimizer update on real text | Passed; learned weights changed |
| Bounded frozen-teacher correction calibration | Passed; RMS `8.763289463692406e-05` |
| Matched shallow-control and PCC-student updates | Passed; learned weights changed |
| Frozen backbone / teacher invariants | Passed |
| One-pass student vs training-tail hidden states | Exact equality on a real 2048-token context |
| Target-permuted control | Blocked: one singleton position/norm bucket |

Real-context settings: pair `(4,16)`, fresh seed 2901, bf16 frozen backbone,
fp32 adapter parameters, microbatch 1, 2048 tokens/context, and 32,768 input tokens
(32,752 causal targets) per global update. Teacher, shallow control, and PCC each
completed one optimizer update. Calibration used those 32,768 input tokens,
not the scientific protocol's 1M-token calibration budget. The control was
rejected before its optimizer step and its weights remained unchanged.

Peak allocated GPU memory was **2.432 GiB**, with **2.980 GiB reserved**, in this
bounded diagnostic. The follow-up completed in 31.45 seconds including loading
and checks: teacher update 6.16 s, calibration 1.37 s, paired student update
11.96 s, one-pass comparison 0.25 s. These are local measurements for this
configuration, not full-run throughput or memory guarantees.

## Concrete blocker

Crossed position/norm bucket **69** contained exactly one target. There were
97 nonempty buckets among 100, over 32,752 eligible targets. The required
within-bucket derangement is impossible for that singleton. The production code
raised `Cannot derange singleton target bucket 69 in update 1` as designed.

The original failure is preserved. A separate diagnostic follow-up repeated
the same deterministic inputs/initialization, recorded all bucket counts, and
finished the independent one-pass/frozen-weight checks. It confirmed the same
blocker; no buckets were merged, no self-target was permitted, and no production
permutation policy or scientific gate was changed.

This uses a teacher trained for one update. It does not establish the incidence
of singleton buckets after the full 610-update teacher training. It does show
that full global batch size alone does not guarantee valid derangement. Resolve
the singleton policy explicitly before relying on the conditional control.

These diagnostics establish local feasibility and implementation behavior.
They do **not** establish that privileged feedback or PCC improves validation NLL.
No full layer screen, full probe, test evaluation, or from-scratch training ran.

## Evidence

Files under `temp/dev-diagnostics-20260923-a01/`:

- `inputs.json`, `download-verification.json`: source revisions, paths, hashes.
- `pretrained-preflight.json`: 42 correctness checks and model fingerprint.
- `real-context-diagnostics-failure.json`: original singleton failure.
- `real-context-followup.json`: completed checks, timings, memory, all bucket counts.
- `run_real_contexts.py`, `run_real_contexts_followup.py`: bounded diagnostic harnesses.
- Matching `.log` files retain process output. Follow-up status is
  `blocked_permutation`, not an all-passed result.
