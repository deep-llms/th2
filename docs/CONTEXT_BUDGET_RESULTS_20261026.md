# Context-budget screen — 26 October 2026

**Preserving local attention is substantially better than removing heads.**
A learned router gives only 4/8 GQA groups full history; the rest attend to the
first four tokens and the last 128 tokens. All heads remain active. The backbone
is frozen. See [design and related work](CONTEXT_BUDGET_PLAN_20261026.md).

## Quality

| Policy | 2k PPL, 256 documents | 8k PPL, 32 documents |
| --- | ---: | ---: |
| Original dense | 18.9674 | 14.9860 |
| Local only | 23.9002 | 20.7497 |
| Fixed groups, 4/8 global | 19.3578 | 15.5134 |
| Fixed layers, 14/28 global | 20.0535 | 16.7018 |
| Periodic layers, 14/28 global | 20.9488 | 17.8559 |
| Joint reconstruction oracle | 19.1107 | 15.1880 |
| Learned joint router | **19.1899** | **15.3083** |
| Learned independent router | 19.2066 | 15.3270 |

The learned joint router increases PPL by **1.17% / 2.15%** versus dense at 2k/8k.
On the same 2k development documents, the previous learned head-removal router
was 23.4644 PPL with 6/8 GQA groups retained, a 23.71% increase. Different interventions
and global budgets: this is a directional comparison, not an equal-latency result.

Joint versus independent targets improve NLL by 0.000869 nats/token at 2k
(paired document-bootstrap 95% CI −0.001120 to −0.000614; 173/256 documents)
and 0.001215 at 8k (CI −0.001714 to −0.000709; 26/32). This is a small additional
gain. It is conditional evidence from one training seed and these document samples,
not uncertainty over seeds or a downstream benchmark result.

Fixed groups select their four groups per layer by mean training joint cost.
Fixed layers select 14 entire global layers by training local-output error divided
by dense output energy. Periodic layers use even global / odd local layers. The
oracle minimizes output reconstruction error on the altered model's current states;
it is not an oracle for LM loss and requires both full and local attention.

## Actual cached-decode speed

Whole-model BF16 native SDPA, eager Hugging Face StaticCache. Means of two GPU
replicates' median latencies; each median uses three 32-step measurements after
warmup. The speedup table averages the two paired speed ratios.

| Batch / context | Dense ms/step | Fixed groups speedup | Fixed layers speedup |
| --- | ---: | ---: | ---: |
| 1 / 2k | 21.68 | 0.806× | 0.996× |
| 8 / 2k | 21.02 | 0.811× | 0.998× |
| 1 / 8k | 21.95 | 0.798× | 0.998× |
| 8 / 8k | 33.02 | **1.142×** | **1.560×** |

At batch 8/context 8k, fixed groups take 28.90 ms/step with a 3.52% PPL increase;
fixed layers take 21.17 ms with an 11.45% increase. Thus faster whole-layer switching
loses noticeably more quality. Fixed groups are slower in the other measured cases:
extra calls and layout operations outweigh saved work there. No general speedup claim.

**The learned router has no measured selective-execution speed yet.** Its quality
reference computes both branches; the static methods really slice KV before SDPA.
All methods retain full KV storage. Measurements include projections, MLPs, logits,
cache updates and policy overhead, but use synthetic teacher-forced tokens and no
CUDA graphs/compiler or optimized serving engine. Prefill cost is not included in
these decode latencies; the chunked local prefill reference is slower than dense.

## Interpretation and next decision

Continue with **local/global context allocation**, not whole-head removal. Before
backbone training, test an efficient execution path for the learned policy against
fixed groups on the intended batched long-context workload. Its smaller PPL gap is
useful only if routing/dispatch overhead does not erase the speed benefit. Keep
fixed groups as the practical control. A cheap retrieval test is also necessary:
natural-text PPL alone cannot certify that distant facts are retained.

This is incremental evidence, not a novel architecture claim. DuoAttention,
All-or-Here, Switch Attention and On-Demand Attention already cover closely related
mechanisms. The observed joint-target advantage is too small by itself to justify
an expensive training sweep. No new job is queued automatically.

## Protocol, checks and saved artifacts

Both router banks used identical initialization and the previous native HF Trainer
recipe: 1,024 training documents, 2,097,152 teacher context tokens, 131,072 cached
feature examples, 512 optimizer updates/eight passes. Only router parameters were
trained. Trainer fitting took 5.9 seconds; feature extraction, numerical checks,
quality evaluation and profile are additional work. The full three-job GPU queue
ran from **19:00:49 to 19:03:33 SGT** (about 2 minutes 45 seconds).

The 2k probe is byte-identical to the previous router pilot: reused development
validation, with the earlier 128 diagnostic documents excluded. The 8k probe
shares **23/32 documents** with the 2k validation set; these are not independent
replications. Neither overlaps the 1,024 fitting documents. The eight extra long
calibration documents selected during preparation were unused. All choices were
frozen on training data before either scope was scored.

74 local tests passed, including actual two-rank native Trainer execution for the
old and new studies. B200 FP32 teacher identity and optimized/static-reference
checks had zero discrepancy; full/cached and prefix-causality maximum absolute
logit error was 0.0001593, below the declared tolerance. BF16 outputs were finite;
trained-router reload and cache checks passed. This does not claim bitwise BF16
full/cached identity, arbitrary generation API support, or tested training resume.

Run `6y3o-efficient-context-budget-20261026-a01`; source/launch `81ef234`, read-only
final verification `768740f`, export `4074c91`. Model/data hashes passed before GPU
reclamation. Accelerate config was copied/verified, and the exact authorized prior
burn workers were stopped before confirming all GPUs free. Burns automatically
restored and verified at **19:04:39 SGT**, independently healthy **19:09:24 SGT**,
with all eight ranks ready and collective payload advancing. Guard released.

Durable evidence:

- [All per-document quality results](results/context_budget_20261026/evaluation.json)
- [Whole-model decode measurements](results/context_budget_20261026/decode-profile.json)
- [Training summary](results/context_budget_20261026/training-summary.json)
- [Selected groups/layers and model identity](results/context_budget_20261026/context-config.json)
- [Artifact checksums](results/context_budget_20261026/artifact-hashes.json)
- [Verification receipt](results/context_budget_20261026/verification.json)

All 13 exported files matched independent remote SHA256s. Full raw downloads,
including the 16 MB learned router file, are in
`/disk/thuat/efficient_llm/temp/context_results_20261026/`.
Raw receipts: `temp/context_launch01.log`, `temp/context_health02.log`.
Native router checkpoints 256/512 with optimizer/scheduler and cached features
remain under the run's `supervised/run/study/` on B200. They are not backed up as
full checkpoints locally. No environments, drivers or backbone weights changed.
