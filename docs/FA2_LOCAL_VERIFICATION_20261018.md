# FA2 stable-release check on the development machine

The user requested testing locally before deciding whether to replace FA4.
FA2 2.8.3.post2 passes the kernel and model checks below on an A100. All twelve
complete-training-update timing runs finished successfully. This is not a B200 comparison,
a production backend change, or approval for a long training run.

## Runtime and isolation

- Local hardware: four A100 PCIe 40 GiB GPUs, driver 525.125.06. Numerical jobs
  used free GPUs 0 and 1; timings use only GPU 0, sequentially.
- Test runtime: Python 3.11.15, PyTorch 2.8.0+cu126, Transformers 5.9.0,
  Accelerate 1.13.0, Datasets 4.8.5, Tokenizers 0.22.2, Safetensors 0.8.0.
  The production CUDA 13 environment cannot run on this development driver.
- Downloaded the official FA2 2.8.3.post2 CUDA 12 / torch 2.8 / Python 3.11 /
  C++11-ABI wheel, checked its release SHA256, and installed it with `--no-deps`
  into `temp/fa2_local_20261018/packages`. The existing environments and driver
  were not modified. The previously created local CUDA-compatible interpreter
  was reused.
- The local probe substitutes FA2's `flash_attn_varlen_func` for the FA4 kernel
  in the existing test process. It preserves document offsets, input casts,
  GQA, causality, scaling, and the rest of the model/trainer. It adapts FA2's
  output tensor to the tuple expected by the FA4 adapter.
- Existing test argument/metadata names still say `fa4` for the packed-layout
  transport. **The executed kernel is FA2**, identified by `actual_kernel`, its
  version and a nonzero call count in every FA2 receipt. SDPA timing controls
  require zero FA2 calls. This is a test-only substitution, not a public FA2
  training option or a resumable production recipe.

No B200 commands, GitHub pushes, Dropbox checks, dataset downloads, process
stops, or production model/default changes were made for this comparison.

## Numerical checks

| Check | Maximum relative L2 output/logit error | Maximum relative L2 gradient error |
|---|---:|---:|
| BF16 FA2 kernel vs FP32 SDPA math | 0.2023% | 0.3077% |
| Native Qwen, checkpointing off/on, vs BF16 SDPA | 0.2968% | 0.5806% |
| Routed B-DOT, hard/soft and reuse off/on, vs BF16 SDPA | 0.3549% | 0.5819% |

Kernel fixtures use head dimension 128, 8 query / 2 KV heads, irregular document
lengths including one token, sixteen 128-token documents, and one 2,048-token
document. The FP32 math reference disables TF32. Preset relative L2 limits are
1% for outputs and 2% for Q/K/V gradients.

Model fixtures reuse the existing native and 12-layer reduced-width B-DOT
probes. Their existing BF16 limits remain 2% for logits and 5% for parameter
gradients. Routed cases use separated diagnostic router biases to compare the
same discrete computation graph; all primary and secondary choices match.
This does not guarantee identical natural routing across backends. Maximum
absolute routed LM-loss difference is 0.000179.

Nonreentrant activation checkpointing, causal masking and document isolation
pass. With routes fixed, changing another document or later tokens leaves the
protected logits bitwise unchanged. Cross-document input gradients are exactly
zero. No tolerance was relaxed and no production initialization was changed.

## Complete-update timings

Two repeats per model/backend completed. Actual HF Trainer updates
include forward, backward, clipping, AdamW, scheduler and routing bookkeeping.
The full 12-layer configs use BF16 activations/FP32 weights, sequence length
1,024, microbatch 2, accumulation 2, and 128-token document fragments. Five
warmup updates precede 30 measured updates per run. Checkpointing is off.
Each pair's backend order reverses on the second repeat. Input and model seeds
are held fixed. Mixed hard routing sets the diagnostic default prior to zero
to exercise off-diagonal routes; production settings remain unchanged.

| Model | SDPA seconds/update | FA2 seconds/update | FA2 throughput gain | SDPA / FA2 peak allocated GiB |
|---|---:|---:|---:|---:|
| Native Qwen | 0.20887 | 0.18781 | +11.22% | 11.31 / 11.17 |
| Matched dense D | 0.23716 | 0.21608 | +9.76% | 11.63 / 11.48 |
| Mixed hard-routed B-DOT | 0.34184 | 0.33843 | +1.01% | 11.87 / 11.73 |

Times are arithmetic means of two 30-update measurements. Throughput gain is
the ratio of mean SDPA time to mean FA2 time minus one, not the percentage
reduction in step time. Peak memory is PyTorch allocated memory, not total
device occupancy. The routed FA2 repeats were 0.33521 and 0.34165 s/update;
SDPA repeats were 0.34118 and 0.34250. The approximately 1% routed advantage
is small relative to this variation and should not be treated as a robust gain.
This suggests that attention is not the main source of routing overhead here.

All 35-update runs completed, with finite reported aggregate losses. Every FA2
run recorded exactly 840 kernel calls (35 updates × 2 microbatches × 12 layers);
every SDPA control recorded zero. The final read-only check verified both used
GPUs free. No workload was stopped.

Timings exclude preprocessing, process startup, evaluation and checkpoint I/O.
They use synthetic data and the reference expert implementation; they do not
validate the pending grouped-expert implementation or establish convergence.

## Interpretation and sources

FA2 is a viable local training-kernel candidate. Our pinned FA4 4.0.0b33 rejects
A100 backward, so this machine cannot provide a direct FA2-versus-FA4 training
comparison; see the [earlier hardware check](BDOT_GPU_VERIFICATION_20261018.md).

The latest stable FA2 release checked is
[2.8.3.post2](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post2).
Its release notes describe maintenance fixes, including backward offset overflow
and CUDA 13 wheel selection, without new kernel features. Its
[tagged build configuration](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/setup.py)
includes SM100 compilation with CUDA 12.8 or newer. The
[official README](https://github.com/Dao-AILab/flash-attention#flashattention-4-cutedsl)
describes FA4 as optimized for Hopper/Blackwell, including B200. These are
compatibility/design facts, not a measured performance ranking for our model.

FA4 already has [B200 numerical and update checks](BDOT_B200_VERIFICATION_20261018.md)
and [eight-rank resume evidence](BDOT_FA4_RESUME_20261018.md) for tested fixtures.
A stable label alone does not establish that switching this project to FA2 is
safer. A B200 decision requires a compatible pinned FA2 build, the same numerical
checks, full-update timings, and resume validation before production use.
For now, keep the already submitted B200 environment specification unchanged.

## Local artifacts

All raw files are in
[/disk/thuat/DOT/temp/fa2_local_20261018](/disk/thuat/DOT/temp/fa2_local_20261018):

- `probe.py`, `benchmark_queue.py`: test-only adapter and bounded sequential queue.
- `release.json`, `wheel_receipt.json`, `environment.json`: source, hash and runtime.
- `kernel.json`, `native.json`, `routed.json` and corresponding model logs.
- `bench_*.json` / `bench_*.log`: individual full-update measurements.
- `benchmark_progress.json`: completed timing jobs; `benchmark_summary.json`
  is written only after the entire queue succeeds.

The [durable combined receipt](results/fa2_local_20261018.json) contains all
numerical records, twelve timing summaries, environment/wheel identity, reported
aggregate losses and source hashes. It validates the expected actual-kernel
identity and measured update count for every run.
