# Native 12-layer baseline — runtime selection and launch

The user authorized stock FA2 with older PyTorch, comparison with production
FA4 and the isolated FA2 compiler-fix build, English-only CulturaX download,
verification, and a fresh native Qwen3 12-layer baseline stopping at 5,000 updates.
The architecture/optimizer defaults come from `BDOT_v2_spec.md`. No router,
FiLM, pretrained weights or additional research arms are included.

## Live baseline verified — 2026-10-20 04:35 UTC

The authorized setup/comparison/download/verification sequence is complete;
the fresh 5,000-update native Qwen3 12-layer baseline is actively training.
Execution root: `/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a02`,
output `prepared/baseline`, supervisor session `native12_train_20261020_a02`.
Implementation/launch commit caea83d; subsequent commits only retrieve status.
**Do not launch it again.** Source/execution commands.sh are returned to #0.

FA4/train selected after stock torch2.8/FA2 and compiler-fixed FA2 comparisons.
All 3,008 English dataset files and six tokenizer/config files pass pinned
SHA256 checks. Caches reused in 8.917 seconds with 29,278,633 train and 9,704
validation sequences. Original train/validation split, EOS, within-document
causality and reset positions are verified. Protected train/eval environments
and the NVIDIA driver remained unchanged.

Corrected eight-rank resume parity passed (weights max abs 3.7253e-9,
optimizer 2.3283e-10; scheduler/eight-rank RNG match). Real-model smoke25 and
resume50 both passed with matching training/data receipts. The first a01 smoke
failed before training due solely to the modern-cuDNN metadata getter; its
outputs remain separate. Only that reader changed; no precision flags changed.

At 04:35:32 UTC, baseline had logged step 340 (loss ~6.228), checkpoint250
complete including optimizer, scheduler, all eight RNG files, and matching
recipe. Validation250: loss 6.60917368, PPL 741.869745 over 9,915,698 scored
tokens/9,704 sequences. GPUs 0–7 each have one training worker, 97–99% utilization,
54,110 MiB device usage; guard held by the supervisor. No run failure observed.

Recipe: seed0, seq1024, micro16 × accumulation4 ×8 GPUs =524,288 input tokens,
full schedule57,220/cutoff5,000/warmup1,145, BF16, tf32False, checkpointingFalse,
AdamW3e-4 cosine floor.1, eval/save250, retain all checkpoints. The supervisor
will restore and verify communicating burns automatically after success/failure.
Estimated completion around 05:30 UTC (roughly one hour total, including eval/
saving; estimate from early observed progress, not a completion promise).

Receipt: `docs/results/native_baseline_launch_20261020.json`; report:
`docs/ATTENTION_BASELINE_LAUNCH_20261020.md`. Raw local evidence in
`temp/attention_study_20261020/`, especially baseline_health01.log and
retry_smoke25.json/retry_smoke50.json. Next status check should read the existing
queue, final baseline-5000.json, supervisor and burn-verified receipts; require
step5000/artifact success and live verified burns before declaring completion.


## Runtime choice

Protected: `train` and `eval` (torch 2.14.1 / CUDA 13 / FA4 4.0.0b33), and
`train_fa2` (same torch, FA2 2.8.3.post2 with C++20 build flags). New isolated
runtime: `train_fa2_t28`, torch 2.8.0 / CUDA 12.8, official stock FA2 wheel.
All eight B200s use driver 580.167.08; no driver/system CUDA change is needed.
The new torch reports compiled SM100 support. Its controller install lacked
the wheel, so the exact official release artifact is requested separately.

Numerical eligibility checks: BF16 varlen forward/backward against FP32 SDPA
math; native model logits/gradients/document isolation with checkpointing on
and off; routed model checks with stable controlled routing. Two full-update
8-GPU timing repeats use the actual 12-layer dimensions, microbatch 16,
accumulation 4, sequence 1,024, and native plus mixed routed models.
FA4 wins ties. A switch requires over 5% native speed gain, disjoint repeat
timing ranges, and no routed regression greater than 2%. These are bounded
checks, not a guarantee of identical long-run trajectories across runtimes.

Selected-runtime gates before the baseline: eight-rank uninterrupted vs
checkpoint/resume parity (weights, optimizer, scheduler, RNG and evaluation),
then a separate real-model 25-step smoke resumed to 50. Any failure stops the
queue and invokes the existing verified burn handoff.

## Data and recipe

`resources/cx_english_20261020.json` pins the dataset/tokenizer revisions and
trusted file hashes. Only `nht10/cx_sampled`, subset
`qwen3_0.6b_base_en_30B`, is requested: 751 train shards, one validation shard,
3,008 data/index/metadata files (~135 GB). Verify every SHA256 and exact saved
Dataset document counts before use. The original fixed validation split is
retained. Tokenizer/config inputs are also independently hash-checked.

Reuse train.py's ordinary HF dataset maps/cache: EOS after every document,
causal attention isolated per document, position reset, fixed seed-0 packed
shuffle. No separate token export. A CPU helper may populate these exact caches
while burns stay active; Trainer calls the same preprocessing functions.

Applied continuation-preserving schedule (no reply to the optional question):
57,220 full updates, force stop 5,000; warmup 1,145 (2%). AdamW, LR 3e-4,
betas .9/.95, weight decay .1, clipping 1, cosine floor .1. Batch: 16 × 4 × 8
× 1,024 = 524,288 input tokens/update, 2.62144B at cutoff. BF16 autocast,
FP32 weights, decoder activation checkpointing disabled. Save and evaluate
every 250 steps; retain every checkpoint. Smoke outputs are separate from the
fresh baseline. The 30B schedule is nominal; packing discards map-batch tails.

## Safe execution

Prior th2 examples inspected: 5a6e9a5, 931839d, 82f6428 and 22e3dc7. Reuse
`scripts/train_then_burn.py`: inspect known burn PID identities, guard ownership,
wait/recheck, signal only verified workers through pidfd, require all eight
GPUs free. `scripts/bdot_b200_launch.py` copies the resource Accelerate YAML
into the actual HF cache, compares bytes and saves `accelerate env` (also in
the selected interpreter). The supervisor cleans only its owned descendants,
then starts and verifies communicating burns on all eight GPUs after success
or failure. No blanket process matching or driver/environment replacement.

Evidence lives in ignored `temp/attention_study_20261020/`. Local regression:
44 ML tests, 7 old-torch CPU adapter tests, four runtime-selection tests pass;
the baseline queue builder's schedule/retention contract also passes. Final GPU and launch evidence is recorded in the live-baseline section above.

Initial B200 evidence: stock wheel installed after published SHA256 validation
at 01:27 UTC. All three environments passed kernel/native/routed numerical
checks. At 01:36 UTC, first native complete-update timing was 0.614386 s (FA4)
versus 0.627193 s (stock FA2/torch 2.8); repeats/selection pending. FA4 peak
allocated memory was 53,841,749,504 bytes (~50.1 GiB). These are full training
updates, excluding dataset preparation, evaluation and checkpoint writes.

Final comparison completed at 01:48:24 UTC: **FA4 selected**. All nine numerical
checks and twelve complete-update timing jobs passed. Mean native seconds/update:
FA4 0.615090, stock FA2/torch 2.8 0.626876, patched FA2 0.610053. Mean routed:
FA4 0.875557, stock FA2 1.482251, patched FA2 0.822259. FA4 routed repeats varied
by about 10%; this comparison does not establish a precise small routed speed
advantage. The patched build fails the required >5% native improvement.
See [compact receipt](results/attention_comparison_20261020.json).
Eight enhanced burn workers and advancing collectives were verified, and the
guard was released. Live PID ownership matched at 01:48:31 UTC. Data download
uses the controller's established single-file #models URL route with exact
file destinations; only English pool files and tokenizer/config inputs are listed.

Data-transfer update: eight disjoint controller batches are active. Direct node
checks are required because the controller's per-task progress lines can lag.
At 02:21 UTC, 463/3,008 files were present at expected size (~21.36 GB); final
SHA256 validation remains pending. With 192 unrestricted CPU cores and ~1.96 TB
available RAM verified, the initial HF map/cache uses 128 processes, fixed in
the recipe. A local CPU data-only check confirmed identical recipes and cache
reuse on the second call. This is not a second token-export format.

## Preparation and smoke correction

All 3,008 dataset hashes, six tokenizer/config hashes, and document counts
passed. Exact HF caches contain 29,278,633 train sequences and 9,704 validation
sequences.

At 04:11 UTC, a01 real-smoke25 failed before the first optimizer update:
the legacy torch.backends.cudnn.allow_tf32 getter raises when HF sets
convolution precision to IEEE independently of RNN precision. This was a
resume-metadata compatibility bug, not a kernel/data/OOM failure. Supervisor
finished at 04:12:27 with all GPUs free before verified communicating burns
93042–93049, guard released. Full local evidence: smoke25-failure.log and
pipeline_status09.log under temp/attention_study_20261020/.

Fix: read public per-operator precision fields on modern torch without changing
any flags; retain legacy reader for older torch. Seven resume-recipe tests pass,
including the exact split-cuDNN-policy reproduction and mismatch rejection.
The GPU parity probe now explicitly uses tf32=False, matching the real recipe.
All data hashes and caches remain valid. Retry uses fresh a02 outputs, retaining
a01 evidence. Its separate burn-predecessor gate requires completed cleanup,
matching verified burn receipts/live ownership and released guard, even though
a01 training failed; runtime comparison still independently must pass. Five
download/handoff tests pass. The same eight-GPU parity, smoke25/resume50 and
fresh baseline5000 stages remain mandatory. No precision/runtime/data change.

Precision API reference: [official PyTorch CUDA notes](https://docs.pytorch.org/docs/2.14/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).
