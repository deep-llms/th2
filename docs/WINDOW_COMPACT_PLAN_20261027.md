# Sink/window and compact scheduling screen

27 October 2026 SGT. User approved testing whether sink-plus-window execution
overhead explains the absence of a whole-model speedup. No training, environment
changes, downloads, or deletion of prior outputs. Reuse the frozen 28-layer
Qwen3-0.6B-Base, fitted routers and calibration-selected fixed group assignments.

## Changes and controls

- Keep the original capacity-sized Triton kernel as a control and SDPA as default.
- Opt-in `triton_compact`: launch full-history tiles only for selected global
  heads; each local head uses one tile and writes its final output directly.
  Only global heads need partial storage/reduction. GPU-side group mapping uses
  the known fixed selection count, without synchronizing the mask to the CPU.
  Full KV still exists. At most 256 local keys; malformed count/mask combinations
  return NaNs rather than reading out of range.
- Compare the same fixed half-global groups with first-four + recent-128 versus
  recent-128 only. Zero-sink SDPA decode uses a direct slice. Allocations were
  selected with four sinks originally; this is a mask ablation, not a retuned
  window-only method. No new router trained for zero sinks.
- Preserve initial-token RoPE positions, causal semantics, cache ownership and
  prior model/router hashes. This scheduling change is inference-only.

## Tests and measurements

1. CPU interpreter tests against SDPA and original kernel: all-local, all-global,
   mixed/per-batch masks, zero/four sinks, overlap, noncontiguous KV, poisoned
   history/padding, malformed counts, and real tiny Qwen cached trajectories.
2. Compiled B200 FP32/BF16 attention tests and original-weight FP32 trajectories.
   Original tolerances: FP32 attention atol 3e-5/rtol 3e-4; BF16 .002/.01;
   FP32 whole-model .002/3e-4. Abort queue on a gate failure and restore burns.
3. Whole-model BF16 eager decoding: batch 1/8 × context 2k/8k, two GPU replicates,
   rotated policy order, three 32-token measurements after warmup. Include native
   dense, original-kernel dense, compact dense, fixed SDPA/original/compact at
   zero/four sinks, and original/compact joint routing at four sinks.
4. Attention-only microbenchmark at those shapes. Capture eight fixed-shape
   attention calls per CUDA graph, three repetitions of 100 replays; report
   this separately from eager whole-model latency. Check capture/eager equality.
   Report dispatched tile/reduction counts. No whole-model graph claim.
5. Eight real documents × 32 cached positions: compare BF16 original/compact
   logits, NLL and predictions. Preserve any differences. The earlier failed
   native BF16 gate remains failed; compact timing is experimental, not acceptance.
6. Native SDPA PPL on the same 256 × 2k development documents: dense, fixed-four
   sinks, fixed-zero sinks. Paired document losses, no new fitting or selection.

## Operation and decision

Use the unchanged supervisor: input hashes → copy/verify Accelerate and env →
fresh exact burn identities → stop only those workers → verify eight GPUs free →
gate → measurements → PPL → automatic communicating burns, including on failure.
Export and verify artifacts, then set commands.sh to #0. No next queue by default.

If compact scheduling helps attention but not total decoding, report the remaining
overhead instead of equating fewer KV reads with speed. If sinks barely affect
timing but protect PPL, retain them. No latency, cache-capacity or novelty claim
until supported by measurements against the fastest tested dense control.
