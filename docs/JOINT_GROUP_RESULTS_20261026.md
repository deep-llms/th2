# Joint-group diagnostic results

Completed **26 October 2026, 09:29:27 SGT**, on all eight B200s. Both the
pretrained-model GPU correctness gate and the distributed study passed.
This is frozen-model evaluation, not router training or a sparse-kernel benchmark.

Protocol: [JOINT_GROUP_DIAGNOSTIC.md](JOINT_GROUP_DIAGNOSTIC.md). Original full
28-layer Qwen3-0.6B-Base; BF16 SDPA. 64 calibration train documents and 128
disjoint validation documents, seed 42; one prefix with 2,048 scored next-token
targets per document (262,144 evaluation tokens). These are long English
document prefixes, not full validation PPL.

| Policy | Keep 6/8: loss | Keep 6/8: PPL | Keep 4/8: loss | Keep 4/8: PPL |
|---|---:|---:|---:|---:|
| Original dense (8/8) | 2.877809 | 17.7753 | 2.877809 | 17.7753 |
| Calibration-selected static | 3.285976 | 26.7351 | 4.967047 | 143.6022 |
| Raw group norm | 3.080002 | 21.7584 | 4.370010 | 79.0444 |
| Projected group norm | 3.054024 | 21.2005 | 4.443120 | 85.0398 |
| Joint subset selection | 3.030319 | 20.7038 | 4.023279 | 55.8841 |

Dense is repeated as a reproducibility check; its per-document losses matched.
All reduced-group policies modify every layer and use their own resulting
hidden states. Static groups are chosen only from calibration data. Norm and
joint policies inspect actual attention outputs, so they are offline references.

## What this says

Joint selection improves next-token loss over both magnitude references, not
just its reconstruction objective. At 6/8 groups it beats projected norms on
all 128 documents. Joint minus projected-norm loss is **−0.023705 nats/token**,
with a paired document-bootstrap 95% interval **[−0.025049, −0.022412]**.
Against raw norms the difference is **−0.049683**, interval
**[−0.051781, −0.047587]**, also improving all 128 documents.

At 4/8, joint minus projected norms is **−0.419840 nats/token**, interval
**[−0.446703, −0.391752]**; 126/128 documents improve. However, the resulting
PPL 55.88 is far above dense 17.78. Even 6/8 remains worse than dense by
**+0.152509 nats/token** (PPL 20.70 versus 17.78), on every evaluated document.

The local-MSE reference is not an oracle for LM loss. At 4/8, projected norms
have lower local reconstruction error than raw norms, but worse LM loss.
Local reconstruction alone therefore cannot decide which policy is useful.

This supports a small next test of whether a cheap causal router can predict
useful joint subsets at **6/8**, against matched magnitude-supervised routers.
It does not establish dense-quality preservation, deployability, speed, cache
savings or novelty. No follow-up training was launched. Bootstrap intervals
describe documents in this fixed probe; they do not measure training-seed variance.

## Correctness and safe execution

- 59 local tests passed: 48 helper/supervisor and 11 mechanism/scoring tests,
  including serial versus uneven two-rank inference.
- Actual pretrained B200 check: keep-all output exactly equals dense; future
  changes do not alter causal prefixes in FP32 or BF16; all K/V groups remain
  cached. Maximum FP32 full-versus-cached logit difference: **0.000212** across
  the five policies on the 24-token GPU check. This does not claim bit-identical
  BF16 cached subset choices across different execution kernels.
- Pinned pretrained weight, tokenizer and consumed dataset-shard hashes passed.
  The saved probe manifest records all document identities and source hashes.
- Accelerate config copied to its actual cache and verified as eight-GPU BF16.
  Exact verified burn workers were stopped, then all eight GPUs were checked free.
- Both jobs exited zero. Communicating burns were restored and verified at
  **09:30:38 SGT**, then independently observed at **09:32:02 SGT**, one worker
  on each GPU, with the guard released. No environment or old outputs were changed.

## Saved evidence

- [Full losses, per-document results, intervals and per-layer reconstruction](results/joint_groups_20261026/diagnostic.json).
- [GPU checks](results/joint_groups_20261026/gpu-check.json),
  [completion and artifact hashes](results/joint_groups_20261026/complete.json).
- [Probe/document provenance](results/joint_groups_20261026/probe-manifest.json),
  [immutable source hashes](results/joint_groups_20261026/source-manifest.json).
- [Supervisor receipt](results/joint_groups_20261026/supervisor.json) and
  [verified eight-GPU burn handoff](results/joint_groups_20261026/burn-verified.json).

Source/launch commit `1f786de`; health `c1b710f`; export `b850d88`.
Downloaded artifact SHA256s and counts were checked locally against completion
receipts. Probe-manifest hash matches the result; source manifest matches the
launched source contract. Raw launch/health logs remain under ignored
`temp/th2_migration_20261026/`.

Remote run root:
`/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-joint-groups-20261026-a01`.
The GPU correctness job took 10.38 s and the distributed study job 34.01 s,
including startup/loading. These are small diagnostic-job durations, not
fair inference-throughput measurements: dense scoring also collected reference
statistics, and the oracle policies deliberately execute all attention.
