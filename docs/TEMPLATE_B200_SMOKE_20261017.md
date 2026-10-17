# Template B200 smoke tests — 17 October 2026

## Scope

Offline synthetic two-layer Qwen3, vocabulary128, hidden size256, four attention
heads/two KV heads, packed length64. BF16 autocast with FP32 parameters,
gradient checkpointing, batch2 per GPU and two accumulation steps.
SDPA uses attention dropout0.1 to exercise RNG restoration; FA4 requires dropout0.
These are correctness smoke tests, not throughput or long-training evidence.

Each workflow trains to step2 of an8-step cosine schedule, saves native state,
resumes to step4 with the same schedule, evaluates seven validation sequences
(433 valid targets), verifies artifacts, then compares with uninterrupted
training to step4. Two-GPU validation uses an uneven seven-sequence split.
Comparisons cover weights, optimizer, scheduler, each rank's RNG, finite gradient
norms, and agreement between periodic and standalone token-weighted validation.
Do not compare workflow losses across backends: their dropout differs. The
separate numerical probe compares identical weights with dropout0.

Runtime: existing eval_fa4 environment on q359, torch2.14.1+cu130,
transformers5.9.0, accelerate1.13.0, datasets4.8.5, flash-attn-4 4.0.0b33.
No package/driver changes. Code snapshot: template_validation in th2;
source_manifest.json records the source hashes. Launch commit1673fb7.

## Runner compatibility

Two earlier submissions were blocked before execution because the runner's
pattern check matched Hub-option references, including a disabled declaration.
The guide already documents this behavior; requiring a controller change was
an incorrect conclusion. Removed the unnecessary Hub-option override and kept
standard HF defaults, forced offline flags, disabled telemetry, and local-only
asset loading. A local test proved an attempted Hub request raises
OfflineModeIsEnabled before HTTP transport. No outbound operation was hidden.
The approved launch contains no publication request.

## Evidence

Remote root: /mnt/local/_outputs/deep-llms_th2/q359-template-smoke-20261017-a03.
Local raw receipts: /disk/thuat/deep2shallow/temp/template-b200-20261017/.
Final structured evidence: [result JSON](results/template_b200_smoke_20261017.json).

## Verified results

| Check | Result | Max resumed/reference weight difference | Optimizer difference |
|---|---|---:|---:|
| SDPA, one GPU | Passed | 0 | 0 |
| SDPA, two GPUs | Passed | 0 | 0 |
| FA4, one GPU | Passed | 0 | 0 |
| FA4, two GPUs | Passed | 0 | 0 |

Scheduler, all-rank RNG, finite gradients, completion contracts, and agreement
between periodic and standalone validation passed in every workflow.

The separate real FA4 probe passed with and without activation checkpointing:
relative logit L2 errors0.00299/0.00273 and gradient L2 errors0.00563/0.00531,
below fixed0.02/0.05 limits. Changing an earlier document left subsequent
document logits exactly unchanged.

Tests finished16:36:50UTC. Supervisor completed successfully and restored all
eight burns16:38:02. A subsequent read-only receipt16:38:27 verified matching
worker ownership, released guard, and advancing collective cycles10 through50.
Runner left inactive (#0). Old research outputs and local unrelated edits
were preserved.
