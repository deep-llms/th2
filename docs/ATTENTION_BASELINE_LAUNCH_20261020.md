# Native 12-layer baseline — runtime selection and launch

The user authorized stock FA2 with older PyTorch, comparison with production
FA4 and the isolated FA2 compiler-fix build, English-only CulturaX download,
verification, and a fresh native Qwen3 12-layer baseline stopping at 5,000 updates.
The architecture/optimizer defaults come from `BDOT_v2_spec.md`. No router,
FiLM, pretrained weights or additional research arms are included.

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

Proposed continuation-preserving schedule (optional user question pending):
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
the baseline queue builder's schedule/retention contract also passes. GPU
results and the actual launch receipt will be recorded when available.

Initial B200 evidence: stock wheel installed after published SHA256 validation
at 01:27 UTC. All three environments passed kernel/native/routed numerical
checks. At 01:36 UTC, first native complete-update timing was 0.614386 s (FA4)
versus 0.627193 s (stock FA2/torch 2.8); repeats/selection pending. FA4 peak
allocated memory was 53,841,749,504 bytes (~50.1 GiB). These are full training
updates, excluding dataset preparation, evaluation and checkpoint writes.
