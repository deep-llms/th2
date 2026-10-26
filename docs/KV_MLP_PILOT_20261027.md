# KV sharing and MLP pruning: bounded interaction pilot

Authorized by the user's “ok do it”, following the research discussion.
This is a diagnostic study of pretrained conversion, not a new pretraining run
or a commitment to a publishable mechanism. Reuse the existing 6Y3O B200 runner,
verified Qwen3-0.6B-Base weights and English CulturaX pool. No node/env changes.

## Question and controls

Does cross-layer KV sharing change the capacity/channels that MLPs need, and
does simple recalibration already account for any interaction?

Keep the original 28-layer model. Fixed adjacent pairs: zero-based layers
0/1 share layer 0's KV, 2/3 share layer 2's, etc. Every layer retains its own
Q, Q-normalization, attention and O projection. Consumers have no K/V projections
or K norm, and store no second copy of KV. Shared tensors retain gradients.
Use native SDPA, normal causal attention, and ordinary HF Trainer/Accelerate.
Decoder activation checkpointing is disabled and explicitly rejected.

MLP pruning physically slices the matching gate/up rows and down columns.
Channel ranking uses calibration activation mean square times down-column
weight squared norm. It ignores channel correlations and is a baseline.
Per-layer widths are 100, 87.5, 75, 62.5 or 50 percent of original width.
Single-layer NLL curves support a greedy marginal allocation retaining exactly
75% of total MLP width. This is not claimed to be an optimal joint allocation.

Six fresh recovery arms, same original initialization and data:

| Arm | KV | MLP |
|---|---|---|
| A | independent | original |
| B | paired sharing | original |
| C | independent | ranks/width allocation calibrated on dense |
| D | paired sharing | exactly C's ranks and allocation |
| E | paired sharing | ranks recalibrated after sharing; C's allocation |
| F | paired sharing | ranks and width allocation recalibrated after sharing |

E separates channel selection from F's capacity allocation. A/B/C/D measure the
factorial loss interaction, before and after standard recovery. E/F are simple
recalibration controls, not a proposed novel method. Also measure per-layer
25%-pruning sensitivity after recovery of A and B, with fresh channel rankings
from the same calibration documents. No policy uses held-out scores.

## Data and recovery budget

Use existing deterministic long-document prefix preparation, seed 8042, a fresh
probe at length 1024. This is deliberately a bounded prefix diagnostic, not
the packed full-corpus training recipe. Each row is one document; therefore no
cross-document attention. Input is tokens[:-1], target tokens[1:], exactly 1024
targets per row. Existing EOS semantics apply at genuine document ends; no
artificial EOS is added inside a truncated prefix.

From the pinned training split choose 128 disjoint calibration documents (64
for ranking, 64 for sensitivity) plus 32,768 separate recovery documents. Choose
256 held-out documents from validation. Record document/token hashes and verify
source/model checksums. The bounded first-shard selection is not a uniform
sample of the full corpus. Calibration and recovery are disjoint.

Each arm: 512 optimizer steps, microbatch 8 per GPU, 8 GPUs, accumulation 1,
1024 input tokens => 33,554,432 tokens. Six arms total 201,326,592 recovery
tokens, excluding calibration/evaluation and frozen teacher computation.
This is a complete short recovery schedule, not a cutoff of the 30B schedule.
AdamW, LR 5e-5, cosine decay, 26 warmup steps, weight decay .01, clip norm 1,
BF16 autocast with FP32 trainable parameters, training/data seed 42.
No early selection of weights; retain checkpoints at steps 128 and 512.

Ordinary loss = 0.5 CE(next token) + 0.5 KL(original teacher || student), T=1.
Original teacher is frozen and evaluated on identical prefixes. No custom
consumer/hidden regression loss. All fixed complete microbatches have identical
target counts; DDP rank means give exact global token weighting. Record actual
sample order per rank and require equality across arms. Evaluate step 0/128/512,
preserving RNG and module modes. Report quality curves and elapsed GPU time.

## Gates, performance, and stop criteria

Before recovery: CPU reference tests for dense identity, shared KV gradient
flow, manual attention reference, full/chunked/cached equivalence, actual cache
storage, physical pruning versus masked-reference outputs/gradients,
save/reload, and CE/KL normalization. GPU gate repeats numerical checks on
original weights, followed by an eight-step distributed Trainer smoke before
the full queue. The CPU integration test uses two ranks and tiny weights.
Custom checkpoints reload through `CompressedQwen`/`load_compressed`, not the
generic AutoModel factory, because per-layer widths and KV sources differ.

Measure native SDPA whole-model prefill/decode with actual pruned weights and
shared DynamicCache, against the same dense backend, separately from training
throughput. Report parameters and allocated cache bytes; never equate cache
capacity saving with proportional speed. Eager timings are not serving results.

Proceed no further automatically. The result may be no interaction, ordinary
recalibration being sufficient, persistent interaction, or poor recoverability.
Paired document confidence intervals do not substitute for independent training
seeds or transfer across models. This pilot does not establish downstream parity,
novelty or publication readiness.

GPU lifecycle uses the existing verified supervisor unchanged: CPU preparation,
copy/verify Accelerate config and `accelerate env`, inspect known burn ownership,
exact PID/start-identity reclamation, verify all eight GPUs free, correctness
gate then queue, and automatic communicating burns on success or failure after
owned children exit. Preserve all outputs and restore commands.sh to #0.
