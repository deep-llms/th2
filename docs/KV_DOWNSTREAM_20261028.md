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

Source `eval/kv_harness.py`, `eval/downstream_suite.py`,
`scripts/check_kv_downstream_adapter.py`, `scripts/launch_kv_downstream.py`.


## Results — complete

Full evaluation finished **01:21:47 SGT, 28 October**. Scores below are percentages; differences are Uniform25 minus dense A, in percentage points.

| Task | Dense A | Uniform25 | Difference |
|---|---:|---:|---:|
| HellaSwag | 53.70 | 51.88 | -1.82 |
| ARC-Easy | 64.27 | 61.57 | -2.69 |
| ARC-Challenge | 38.05 | 35.24 | -2.82 |
| PIQA | 70.51 | 69.59 | -0.92 |
| WinoGrande | 59.59 | 58.17 | -1.42 |
| BoolQ | 69.17 | 65.17 | -4.01 |
| OpenBookQA | 36.60 | 33.80 | -2.80 |
| SciQ | 92.90 | 92.00 | -0.90 |
| **Eight-task average** | **60.60** | **58.43** | **-2.17** |
| LAMBADA final-word accuracy (separate) | 52.28 | 50.26 | -2.02 |

LAMBADA perplexity: **10.2766 → 11.6019**. The earlier CulturaX prefix PPL was
17.6572 → 18.2008; those metrics measure different token populations.

Uniform25 is lower on all nine tasks. The eight-task mean difference is
−2.173 points, with a paired, within-task example-bootstrap95%CI
**[−2.665, −1.683] points** (20,000 draws, seed42, equal task weights). This
conditions on these checkpoints and examples; it does not estimate training-seed
variance or establish a formal equivalence margin. Some individual task intervals
include zero. The aggregate degradation is broader than the PPL probe alone
shows; do not describe this conversion as matching dense downstream quality.

The model has25%fewer KV-producing layers. Actual serving memory/latency/throughput
benefits have not been measured for these long-recovery checkpoints. Evaluation
wall time is not an inference-efficiency comparison.

## Verification and artifacts

- Completed all26,618examples per checkpoint, canonical zero-shot task settings.
  Paired prompt/target hashes, tokenizer, benchmark manifest, weight hashes and
  runtime match. Two likelihood requests per arm were left-truncated under the
  standard1,024-token harness context policy; all other requests fit.
- Real GPU gate: direct native logits and manual continuation likelihood match
  the adapter exactly for both checkpoints. Gate9.98s; eight-example/task
  smoke24.10s; full parallel evaluation150.27s. All stages exit0.
- Accelerate resource config copied/verified8GPU BF16. Only known burn workers
  reclaimed;30second wait and all8GPUs free before evaluation. Communicating
  burns restored/verified01:22:58SGT and independently inspected01:23:02SGT.
  Eight workers630628–630635, launcher630559; guard released, collective progress
  to20cycles/22.21GiB per rank. These are historical identities: reverify later.
- Initial launch a2ec3c6 failed before output creation or GPU stop: launcher
  compared a tokenizer manifest metadata dictionary with a bare digest. Corrected
  bytes/sha256 validation and regression test; fresh a02 launch dc7c0ba succeeded.
  Training weights, model source, datasets, environments and driver unchanged.
- lm-eval prints a harmless local Git-metadata warning because immutable node
  source snapshots omit `.git`. All workers returned0; source identity is
  independently recorded by SHA256 manifests.

Run: `/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-downstream-20261028-a02`.
Full per-example JSON files remain under `supervised/run/full/{A,B}/`. Their
byte/SHA receipts are archived locally. No further training/evaluation queued.

Local results: [full summary](results/kv_downstream_20261028/final/supervised/run/full/summary.json),
[paired accuracy counts and uncertainty](results/kv_downstream_20261028/final/paired-accuracy-analysis.json),
[artifact receipts](results/kv_downstream_20261028/final/artifact-receipts.json),
[full per-example file receipts](results/kv_downstream_20261028/final/full_result_receipts.json).
Raw node log: `temp/kv_downstream_health01.log`, SHA256
`48a696f89032f4b4d6fbf8e5a8488fcf8df7fa85b35f73715048ef835c865782`.
Exported artifacts were checked against node receipts and completion-marker hashes.
