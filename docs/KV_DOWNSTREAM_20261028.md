# Downstream comparison of long KV recovery

28 October 2026, Singapore time. User authorized downstream evaluation after
the completed 1.36B-token dense/uniform25 pair. No training or new model/data
conversion. Use both final step-20,752 checkpoints from
`6y3o-efficient-kv-long-20261027-a01/supervised/run/{A,B}/final`.

Reuse the earlier project's tested canonical lm-eval 0.4.13 tasks and pinned
local benchmark snapshots. The custom loader is `load_compressed`, preserving
each checkpoint's trained KV-source map and MLP widths. Native SDPA, BF16 weights
(same precision as final reported PPL), FP32 log-softmax, seed42, zero-shot,
no chat template, 1,024-token context, batch8. Ordinary right-padded causal rows;
EOS inside a prompt does not split attention. Record normal harness truncations.

| Task | Evaluation examples | Primary metric |
|---|---:|---|
| HellaSwag | 10,042 | acc_norm |
| ARC-Easy | 2,376 | acc_norm |
| ARC-Challenge | 1,172 | acc_norm |
| PIQA | 1,838 | acc_norm |
| WinoGrande | 1,267 | acc |
| BoolQ | 3,270 | acc |
| OpenBookQA | 500 | acc_norm |
| SciQ | 1,000 | acc |
| LAMBADA OpenAI | 5,153 | final-word acc and perplexity |

Full26,618examples per checkpoint. Report every task, an unweighted mean of the
eight multiple-choice tasks, and LAMBADA separately. This is a base-model screen,
not a replication of SwiftKV's instruction-model seven-task evaluation. A small
accuracy gap does not establish equivalence or an inference-speed benefit.

Implementation reuses DOT's `benchmark_data.py` and worker/collector logic,
adapting only the model/precision/provenance checks. Eight independent workers:
four task groups per model, each one GPU and one model load. Per-example outputs
are retained; collector verifies exact populations and paired prompt/target
hashes, tokenizer, dataset manifest, runtime and unchanged weight identity.

CPU verification: native dense harness parity (including empty context and
truncation), shared-model reload/padding/manual likelihood and immutable weights;
all nine real task templates run on tiny local weights; collector rejects changed
prompts, missing populations, wrong protocol and checkpoint hash. Existing GPU
supervisor tests pass. Real-checkpoint GPU logit/likelihood gate and eight-example
per-task eight-GPU smoke must pass before the full screen.

Launcher verifies parent artifacts, source and benchmark hashes before reclaim.
It copies resource Accelerate config to its actual cache and verifies
`accelerate env`. Evaluation itself is eight independent processes, not DDP.
The unchanged supervisor verifies exact known burn worker identities, stops only
those workers, waits30seconds and requires all eight GPUs free. It restores and
verifies communicating burns after success/failure. Earlier outputs, checkpoints,
environments and drivers remain untouched.

Results: pending. Source `eval/kv_harness.py`, `eval/downstream_suite.py`,
`scripts/check_kv_downstream_adapter.py`, `scripts/launch_kv_downstream.py`.
