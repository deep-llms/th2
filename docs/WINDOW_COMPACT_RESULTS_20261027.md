# Sink/window and compact scheduling results

Completed **27 October 2026, 01:19:46 SGT**, on B200. Frozen original 28-layer
Qwen3-0.6B-Base; no training, installation, driver change, or previous-output cleanup.
[Protocol](WINDOW_COMPACT_PLAN_20261027.md).

**Keep the initial four tokens for this frozen-model policy.** Removing them
severely hurts PPL and scarcely affects the custom kernel's attention time.
Compact scheduling removes real wasted work and improves larger attention
microbenchmarks, but does not improve whole-model eager decoding. SDPA remains
the default; both custom implementations remain experimental.

## Plain window versus sink-plus-window quality

Native SDPA, 256 identical development documents × 2,048 scored targets = 524,288
tokens. Same calibration-selected four global groups per layer. Other groups use
either first-four + recent-128 or recent-128 only. All heads remain active, every
layer is modified, and the backbone is frozen. No router or allocation refitting.

| Policy | NLL, nats/token ↓ | PPL ↓ |
|---|---:|---:|
| Dense original | 2.942692 | 18.9668 |
| Fixed half-global, first 4 + recent 128 | 2.963056 | 19.3570 |
| Same groups, recent 128 only | 4.324348 | 75.5163 |

Removing sinks worsens all 256 documents. Paired mean NLL increase **1.361293**,
document-bootstrap 95% interval **[1.330133, 1.393875]**. This supports retaining
sinks in this frozen Qwen intervention. It does not establish that four is optimal,
or that models trained with sliding windows cannot work. The zero-sink variant
uses 128 rather than 132 keys and inherits allocations calibrated with sinks.
Neither condition is a trained window-only architecture.

## Whole-model decode latency

Milliseconds per token step; lower is better. Mean of two GPU replicates' median,
each with three 32-step measurements after warmup, policy order rotated. Full
model includes projections, MLPs, logits, cache updates, router/Python overhead.
BF16 eager execution; synthetic teacher-forced tokens. Prefill and compilation
are excluded. Each policy gets its own fresh StaticCache. Compare within this
run; do not mix these timings with earlier sessions' baseline values.

| Method | B1 / 2k | B8 / 2k | B1 / 8k | B8 / 8k |
|---|---:|---:|---:|---:|
| Dense SDPA | 20.943 | 23.404 | 21.881 | 32.728 |
| Dense original kernel | **20.579** | **22.867** | **21.353** | **22.346** |
| Dense compact kernel | 21.015 | 23.089 | 21.659 | 22.654 |
| Fixed SDPA, 4 sinks | 25.917 | 28.748 | 27.200 | 28.789 |
| Fixed SDPA, 0 sinks | 24.211 | 26.865 | 25.976 | 28.737 |
| Fixed original kernel, 4 sinks | 20.494 | 22.892 | 21.476 | 22.489 |
| Fixed original kernel, 0 sinks | 20.485 | 22.900 | 21.468 | 22.381 |
| Fixed compact kernel, 4 sinks | 20.810 | 23.308 | 21.694 | 22.656 |
| Fixed compact kernel, 0 sinks | 21.428 | 23.118 | 21.640 | 22.675 |
| Joint router, original kernel | 23.734 | 26.463 | 24.660 | 26.179 |
| Joint router, compact kernel | 24.113 | 26.821 | 25.033 | 27.160 |

No material selective-policy advantage over the fastest dense control. At B8/8k,
compact fixed is 0.17 ms slower than original fixed; compact joint is 0.98 ms
slower than original joint. Small differences are not presented as population
confidence intervals. Removing sinks saves some reference SDPA overhead at smaller
shapes (the zero-sink path uses a view instead of concatenation), but does not solve
the overall efficiency problem and incurs the quality penalty above.

## Attention-only execution

Microseconds per attention call, BF16. Eight fixed-shape calls per CUDA graph,
three measurements of 100 replays; capture/eager outputs checked equal. These
numbers reduce Python dispatch overhead for the attention microbenchmark only.
They are **not** whole-model graph decoding or prefill timings.

| Method | B1 / 2k | B8 / 2k | B1 / 8k | B8 / 8k |
|---|---:|---:|---:|---:|
| Dense original kernel | 6.418 | 21.290 | 14.629 | 86.120 |
| Dense compact kernel | 6.916 | 22.075 | 14.593 | 86.123 |
| Fixed original, 4 sinks | 6.405 | 15.206 | 10.812 | 51.496 |
| Fixed original, 0 sinks | 6.409 | 15.258 | 10.789 | 51.575 |
| Fixed compact, 4 sinks | 6.666 | 13.824 | 9.231 | 48.573 |
| Fixed compact, 0 sinks | 6.665 | 13.824 | 9.332 | 48.521 |

At B8/8k, compact scheduling lowers the fixed-policy attention latency by **5.7%**.
It is also 43.6% below dense original attention in this microbenchmark. Sink removal
has negligible effect. At B1/2k the compact mapping overhead makes attention slower.

For the 8k microbenchmark (cache capacity 8,228), the original kernel launches
528 partial programs per batch item plus 16 merge programs. Compact half-global
uses 264 partial/local programs plus 8 merges. Each local head now writes output
directly, while full attention is split only over valid history. This changes
scheduling and intermediates, not persistent KV capacity. GPU-side subset mapping
also costs work. The saved ~2.9 microseconds per layer versus original fixed is
only ~0.08 ms across 28 layers before other overhead; it is not a large whole-model
speed opportunity by itself. Actual eager latency did not improve.

## Correctness and limits

- 82 local tests passed: 29 mechanism tests (including real two-rank fits) and
  53 utility/ownership/manifest tests. Independent dense-mask tests cover the
  sliced zero/four-sink attention semantics. Interpreter tests cover compact
  budgets, noncontiguous KV, overlapping ranges, NaNs and cached model execution.
- 60 compiled B200 attention cases passed. FP32 maximum error versus math
  reference 8.35e-7, versus original kernel 2.98e-8. BF16 maximum versus rounded
  FP32 reference .001953125; original versus compact was exactly equal on these
  sampled attention operands. Poisoned history and invalid-budget checks passed.
- Original-model FP32 cached maximum logit errors: dense 1.66e-5, fixed-four
  1.72e-5, fixed-zero 6.25e-5, joint 0.0, all inside declared tolerances.
- BF16 whole-model comparisons on eight documents × 32 positions are below.
  Independent prefills and changed arithmetic can affect trajectories; this
  experiment does not isolate every source of variation. Raw-attention equality
  on sampled inputs does not guarantee complete BF16 trajectory equality.

| Original → compact | Mean NLL delta | Worst absolute document delta | Changed predictions / 256 |
|---|---:|---:|---:|
| Dense | −0.001472 | 0.012541 | 5 |
| Fixed, 4 sinks | +0.001837 | 0.010478 | 1 |
| Fixed, 0 sinks | −0.002973 | 0.011389 | 3 |
| Joint, 4 sinks | −0.002074 | 0.013591 | 4 |

The earlier .01-nat-per-document native gate remains failed. Even these
original/compact comparisons contain individual documents above that threshold.
The profile explicitly records `diagnostic_only=true`, `native_backend_accepted=false`.
Job success means measurement completion, not qualification of a new BF16 backend.
PPL above uses established native SDPA, not either custom kernel.

## Decision and evidence

Retain sink-plus-window for the frozen model. Keep compact execution opt-in; it
does not earn adoption on whole-model speed or BF16 qualification. This test
rules against the four extra keys being the main cost of the custom attention
path in these shapes. It identifies a real scheduling inefficiency, but removing
it alone does not deliver a useful complete-decoder speedup. No new training is
queued. SDPA remains the default, full KV remains stored.

Run `6y3o-efficient-window-compact-20261027-a01`, source/launch `36e30fa`, independent
health `b10b860`, export `4cf8964`. Gate 54.5s, profile 129.6s, PPL 33.5s, all zero
exit by **01:19:46 SGT**. Inputs verified; Accelerate copied/byte-verified and
eight-GPU BF16 env confirmed; only previous burn workers 602209–602216 stopped,
all GPUs verified free. Environments and previous results untouched.

Automatic burns verified **01:20:56 SGT**, independently observed **01:21:17 SGT**:
eight ranks, collective sum 36, progress to 40 cycles / 44.41 GiB per rank, guard
released. Workers 604193–604200 under launcher 604124 at that observation; recheck
identities before any later stop. Source is this run's immutable
`source/resources/llm_pretrain_burn.py` with the unchanged reviewed burn hash.

Durable artifacts: [gate](results/window_compact_20261027/gate.json),
[all timings and numerical comparisons](results/window_compact_20261027/profile.json),
[per-document quality](results/window_compact_20261027/quality.json),
[completion](results/window_compact_20261027/complete.json), and
[burn receipt](results/window_compact_20261027/burn-verified.json).
Raw local logs: `temp/window_launch01.log`, `temp/window_health01.log`.
Downloaded artifacts: `temp/window_results_20261027/`.
All 11 exports matched independent node SHA256s. Local verification also checked
source/input identity, 88 timing rows, 48 microbenchmark rows, finite metrics,
exact document coverage and burn/completion receipts. See
[verification](results/window_compact_20261027/verification.json) and
[artifact hashes](results/window_compact_20261027/artifact-hashes.json).
commands.sh is inactive #0; no follow-up queue.
