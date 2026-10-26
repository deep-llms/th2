# KV sharing × MLP pruning pilot — 27 October 2026

**The tested recalibration did not help.** Direct adjacent-layer KV sharing causes
most of the quality damage; ordinary distillation recovers much of it, but the
combined models remain clearly worse than dense. Do not scale this conversion
recipe or claim a novel beneficial interaction from this pilot.

## What was tested

Original pretrained **28-layer Qwen3-0.6B-Base**, eight B200s, native SDPA.
Consumers reuse the preceding layer's K/V while keeping their own Q and output
projection. Only the 14 producer layers store KV. MLP pruning physically removes
matching SwiGLU channels, retaining 75% of aggregate width. Per-layer widths vary.
Channel importance is activation mean-square × down-projection column norm-square;
a greedy allocation uses single-layer pruning loss curves. [Design](KV_MLP_PILOT_20261027.md).

| Arm | Change |
|---|---|
| A | Dense control |
| B | Adjacent-pair KV sharing |
| C | MLP pruning calibrated on dense |
| D | B plus exactly C's channels and widths |
| E | D's widths, channels reranked after KV sharing |
| F | Channels and widths both recalibrated after KV sharing |

All six start independently from the original pretrained weights and receive the
same **512 recovery steps / 33,554,432 input tokens**, identical actual sample
order, and loss `0.5 CE + 0.5 KL(original teacher || student)`. LR 5e-5, cosine
schedule, 26 warmup steps, seed 42. Total recovery: **201,326,592 tokens**, excluding
teacher computation, calibration and evaluation. This is a complete short
recovery schedule, not a cutoff of the previous 30B-token training schedule.

Data: 32,768 recovery documents; 64 separate ranking and 64 sensitivity documents;
256 held-out validation documents, each scored on 1,024 next-token targets.
Every sequence is from one document. This is a bounded English prefix probe,
**not full-corpus PPL or downstream evaluation**; the source scan uses the first
32 shards rather than a uniform full-pool sample. These PPLs are not directly comparable
to earlier 2k-context probes.

## Quality

FP32 parameters with BF16 autocast, the training-validation configuration:

| Arm | Initial NLL | PPL at step 128 | NLL at step 512 | PPL at step 512 |
|---|---:|---:|---:|---:|
| A | 3.0231 | 20.74 | 2.997626 | 20.04 |
| B | 13.7414 | 31.43 | 3.266698 | 26.22 |
| C | 3.4148 | 22.81 | 3.076185 | 21.68 |
| D | 14.5258 | 35.18 | 3.356803 | 28.70 |
| E | 16.2799 | 38.73 | 3.426816 | 30.78 |
| F | 15.0049 | 42.22 | 3.481091 | 32.50 |

The large initial sharing losses matter: E/F calibrate on a severely disrupted
model before recovery. Their failure therefore does **not** establish that
calibration after a successfully recovered KV-sharing conversion would fail.
Likewise, one short recovery budget does not establish the best achievable
quality of cross-layer KV sharing generally.

BF16-weight inference quality, matching the timing configuration:

| Arm | PPL | Parameters (M) | KV storage relative to A |
|---|---:|---:|---:|
| A | 20.03850 | 596.050 | 100% |
| B | 26.22722 | 566.688 | 50% |
| C | 21.67890 | 529.990 | 100% |
| D | 28.70302 | 500.628 | 50% |
| E | 30.78658 | 500.628 | 50% |
| F | 32.50281 | 500.628 | 50% |

Casting trained parameters to BF16 changes average NLL by at most +0.000263,
far smaller than the gaps between arms; the ranking is unchanged. This verifies
precision alignment for this held-out probe, not arbitrary backend equivalence.

## Interaction and layer sensitivity

Positive NLL differences mean worse quality. Paired bootstrap over the same 256
validation documents, 10,000 resamples:

| Step-512 comparison | Nats/token | 95% document CI |
|---|---:|---:|
| Interaction `D − B − C + A` | +0.011546 | [+0.006544, +0.018057] |
| Rerank channels `E − D` | +0.070013 | [+0.063451, +0.075610] |
| Reallocate widths `F − E` | +0.054275 | [+0.051291, +0.057294] |
| Total recalibration `F − D` | +0.124288 | [+0.117555, +0.130379] |

The combination has a small adverse interaction beyond adding the two individual
loss penalties. It is about 3.2% of D's total NLL gap to A; most damage comes from
the individual conversions. This is not evidence of useful complementarity.
E beats D on only 7/256 documents, and F beats E on only 3/256.

After recovering A and B, rerank channels separately and prune **one layer at a
time by 25%**, scoring on the fixed 64 sensitivity documents. Layer sensitivity
correlation is **Pearson 0.985 / Spearman 0.922**. Average single-layer NLL penalty
is 0.009401 for A versus 0.009883 for B. The two most sensitive layers remain
27 and 26 in both models (zero-based). There are local differences, but no broad
reordering of MLP capacity needs in this test. These are single-layer ablations,
not optimal simultaneous allocations; A/B also have different recovered weights.

## Measured inference cost

Whole-model eager native SDPA, BF16 weights, DynamicCache. Two GPUs per batch/
context case, three timed repetitions per GPU, rotated arm order. Table values
are medians across the two per-GPU medians; speedups pair arms on the same GPU.
At **batch 8, context 8192**:

| Arm | Prefill ms | Decode ms/step | Mean paired decode speedup |
|---|---:|---:|---:|
| A | 303.84 | 18.390 | 1.000× |
| B | 262.57 | 15.706 | 1.171× |
| C | 297.54 | 18.151 | 1.013× |
| D | 256.29 | 15.821 | 1.162× |
| E | 256.31 | 15.954 | 1.153× |
| F | 255.98 | 15.720 | 1.170× |

D's two paired speedups are 1.145× and 1.180×. Its inference PPL is 43.2% above
A, for roughly 16% higher decode throughput here. Adding MLP pruning gives little
extra decode benefit over KV sharing alone. C's roughly 1% gain is too small to
present as a robust improvement. Smaller-case results vary more; all 48 timing
rows and repetitions are retained. No serving-engine or long-generation claim.
Synthetic input IDs establish cost only, not quality.

Measured KV allocation after the 8192-token prefill, 4 warmup decode tokens and
24 timed decode tokens is **7192.5 MiB for A/C versus 3596.25 MiB for B/D/E/F**
at batch 8. This is exactly half the cache storage, not half the attention work:
all 28 layers still attend. Quality above was measured at 1024-token context;
**8k-context quality has not been established**.

The successful GPU queue, including gates, calibration, six recoveries and
profiling, took 18m49s. Six Trainer calls totalled 12m27s; those durations include
periodic evaluation/checkpointing but exclude setup/post-recovery sensitivity.
The frozen teacher adds compute; native Trainer's FLOP counter is not a complete
accounting of this study.

## Correctness and limits

Native HF Trainer/Accelerate; FP32 trainable weights and BF16 autocast. Shared KV
retains gradients. Eight CPU model/training tests and 55 infrastructure checks
passed, including distributed save/reload. The actual-weight GPU gate and an
eight-step, eight-GPU smoke passed before the six arms. Every arm checked exact
one-pass training-data coverage, matched per-rank order, frozen teacher gradients,
and final checkpoint reload (maximum checked NLL difference zero).

Two earlier attempts stopped in the numerical gate before any training. The
explicit-math reference was corrected to avoid an unintended BF16 QK roundoff,
and the gate now separates implementation correctness from cross-kernel BF16
rounding. Final checks: FP32 full-model explicit-reference relative output/gradient
errors 4.53e-5/9.91e-5; independent BF16 decoder using the same SDPA output error
zero; fixed-input explicit BF16 attention maximum output/gradient errors
0.001873/0.012473 across all 28 layers. Full-model BF16 explicit-math versus
SDPA errors remain large (0.399/0.681): **cross-kernel interchangeability is not
accepted**. Production attention and training were unchanged.

One model, one training seed, a small prefix probe and short recovery budget.
Document confidence intervals do not measure training-seed variation. No broad
quality parity, long-context quality, serving speed or publication novelty is
established. No further training is queued.

## Reproduction and retained evidence

Implementation: `efficient_llm/kv_mlp.py`; calibration/Trainer/evaluation:
`efficient_llm/kv_mlp_study.py`; independent numerical gates:
`efficient_llm/kv_mlp_check.py`; native whole-model timings:
`efficient_llm/kv_mlp_profile.py`. The existing GPU lifecycle supervisor is
unchanged. Run factories are in `scripts/launch_group_diagnostic.py`.

Training source `7abbd37`, run `6y3o-efficient-kv-mlp-20261027-a03`, completed
**05:26:25 SGT**. Evaluation-only source `c140449`, run
`6y3o-efficient-kv-mlp-inference-20261027-a01`, completed **05:39:00 SGT**.
All eight communicating burns restored **05:40:11**, independently verified
**05:42:22 SGT**, guard released. No environment or driver changes.

On B200, both run roots are under `/mnt/local/_outputs/deep-llms_th2/`.
Training checkpoints are under the first root's
`supervised/run/{A,B,C,D,E,F}/trainer/checkpoint-{128,512}`; standalone final
weights are in each arm's sibling `final/` directory.
Large weights/optimizer states remain on temporary node storage, not backed up
by the small-result export. Raw receipts: local `temp/kv_mlp_final01.log` and
`temp/kv_mlp_inference_final01.log`. Failed earlier gate attempts remain preserved.

All **29 exported files** were downloaded and SHA256-verified against node-side
receipts. Full local archive: [`temp/kv_mlp_results_20261027`](../temp/kv_mlp_results_20261027/).
Durable small artifacts: [training summary](results/kv_mlp_20261027/training/summary.json),
[inference quality](results/kv_mlp_20261027/inference/inference-quality.json),
[raw timing records](results/kv_mlp_20261027/training/profile.json),
[derived sensitivity/timing analysis](results/kv_mlp_20261027/analysis.json), and
[final GPU inspection](results/kv_mlp_20261027/final_gpu_inspection.json).
The full 6 MB channel plan and 10 MB document manifest are in the local archive;
compact metadata and their hashes are retained in Git. Exact-file export `8809832`
succeeded after folder export `59e30f9` returned no payloads. commands.sh is #0.

**Decision:** retain this as a negative pilot. There is no demonstrated useful
joint benefit or strong capacity-reallocation signal. If revisited, distinguish
recovery of sharing from calibration of pruning before spending on larger sweeps;
that follow-up has not been run or queued.
