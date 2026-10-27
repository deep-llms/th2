# Final bounded recovery test: Q/K/V-only adaptation

The best fixed layout, uniform25, retains 75% of the KV cache but has a 6.34%
PPL penalty after matched recovery. All three layouts fail the 5% screen. Test
one remaining optimization hypothesis: restricting updates may repair attention
while preserving the rest of the pretrained model better than updating everything.

[SwiftKV, section 5.3/Table 5(b)](https://aclanthology.org/2025.emnlp-main.1306.pdf)
reports better instruct-benchmark accuracy when updating only affected Q/K/V
projections than all later-layer parameters. Its transformation, models, data
and recipe differ from ours. This motivates a cheap control, not a claim that
freezing will improve our base-model PPL or that this is a SwiftKV reproduction.
Neither the layout nor restricted tuning is novel by itself.

## Exact comparison

Repeat uniform25 from the same original dense A-512 parent. Only the parameter
update scope changes. Train Q/K/V linear projections in producer/consumer pairs
(2,3), (6,7), (10,11), (14,15), (18,19), (22,23), (26,27), zero-based. Consumers
already have no K/V projections, so only their Q projection trains. Producers
train Q, K and V. Freeze every other parameter: embeddings/LM head, MLPs,
attention output projections, all norms and unaffected layers' projections.
Gradients still flow through frozen operations to the selected projections.

Keep the same 2,048 steps, 134.22M tokens, per-rank order, seed 42, batch 8/GPU
on eight GPUs, length 1024, LR 5e-5, 102-step warmup, cosine, AdamW, clipping,
weight decay, original frozen teacher and 0.5 CE + 0.5 KL. No fresh data sampling,
new environment/backend, modified attention or custom kernel. Keep all 256-step
checkpoints. This is a new phase from A-512, not a continuation of trained uniform25.

Compare against completed all-parameter uniform25 and matched dense A. Require
identical converted step-zero scores, architecture, parent/model/probe hashes,
actual sample order and recipe. Record the exact trainable names/count and
before/after SHA256 of frozen and trainable parameter bytes. Frozen bytes must
stay identical; some selected parameters must change. Parent and reload checks
remain mandatory. The runtime and already-passed numerical architecture gate
are reused; a new eight-step/eight-GPU smoke gates the real fit.

CPU tests cover scope selection, unchanged initial output, gradient flow,
independent exact frozen-tensor comparison, two-rank training and summary
rejection of altered frozen weights. The existing supervisor verifies the
previous queue/burn handoff, copies/checks Accelerate, freshly identifies known
burn workers, confirms free GPUs and restores communicating burns on completion
or failure. Expect roughly 10–15 minutes including smoke/checks, not a guarantee.

This consumes the fourth and final authorized additional training job. Do not
start another training job afterward. If it fails the quality screen, archive
and stop this recovery branch. If it passes, only bounded evaluation of actual
latency/cache and longer-context quality is justified before claiming utility.
The 5% gate remains upper 95% paired-document NLL CI <= log(1.05); reused
validation and one seed limit the conclusion even on success.

## Outcome

Completed 27 October, 09:36:38 SGT; communicating burns restored and verified.
Q/K/V-only PPL 23.0779, versus 20.4610 for its all-parameter control and 19.2416
for dense A. Correctness checks pass; quality gate fails. No further training.
See [final matched results](KV_RECOVERY_FINAL_RESULTS_20261027.md).
