# Supervised downstream study: A, P6, P7-simple

The initial study ran these three arms (P6, not P6-iso), starting from their
existing seed-42, step-2500 FA4 checkpoints. This is task-specific supervised
adaptation, separate from the completed zero-shot screen.

## Fixed protocol

- English PAWS-X: full training split, 3 epochs; development split for selection;
  final test split has 2,000 examples.
- English NLI: the 392,702 English MultiNLI training examples distributed in the
  pinned facebook/xnli English snapshot, 1 epoch; XNLI development for selection;
  final XNLI test has 5,010 examples. Earlier zero-shot NLI used development.
- Remove training sentence pairs that exactly match development/test text after
  whitespace normalization (and swapped pairs for PAWS). Record removed counts
  and ordered training hashes. No test labels influence preprocessing/selection.
- Fresh 2-/3-class linear head, initialized identically across arms for each
  fine-tuning seed. Full task-connected backbone/consumer/predictor parameters
  are trainable. No new language model and no LoRA.
- Pair format: `Sentence 1: ...\nSentence 2: ...\nRelationship:` followed by EOS.
  Classifier reads the final EOS hidden state. Both sentences are one causal
  document; isolated right padding has reset positions and cannot affect it.
  Longest-first sentence truncation to 512 tokens, including template/EOS.
- Standard HF Trainer and Accelerate, all eight B200 GPUs per run; BF16 autocast,
  FP32 master parameters, FA4; microbatch 16, GAS 1, global batch 128 examples.
  Evaluation microbatch 32. No activation checkpointing.
- AdamW fused, weight decay .01, linear LR decay, 6% warmup, clip norm 1.
  LR candidates 1e-5 and 3e-5; seed 42 for both, per arm/task. Choose using
  development accuracy only, lower LR on a tie; best epoch within each run.
  Repeat the selected LR from the original pretrained weights with seeds 43/44.
- 24 production fitting runs: 12 search + 12 confirmation. The 18 selected
  models (3 arms × 2 tasks × 3 seeds) receive final test evaluation only after
  every fit/selection completes. Report per-task mean/std and per-seed scores.
  Seed 42 participates in LR selection; all arms receive the same search budget.
- Save model-only best checkpoints, Trainer logs, hashes and per-example test
  logits. This is a fresh adaptation study; it does not resume pretraining's
  optimizer/schedule. Model-only snapshots are not optimizer-resume checkpoints.

## Necessary custom behavior

`eval/` previously supported likelihood/perplexity inference only.
`eval/finetune.py` reuses its strict pretrained checkpoint loader and local
hashed benchmark manifest, plus the standard HF Trainer loop. The shared
nn.Module tied-weight saving helper is factored out of `DeepKVTrainer`; its
saving behavior is unchanged.

Task fine-tuning uses only classification cross-entropy, no pretraining proxy
auxiliary loss and no target-normalization updates. P6 already permits task
loss gradients through its estimator. P7-simple opts into task gradients through
its source/estimated memory; this removes its pretraining detach boundaries
only in the new downstream wrapper. Forward values and attention architecture
are unchanged. All other uses default to the previous detached behavior.
Gradient checks require every trainable parameter to have finite gradients,
including predictors, consumers and classifier. An untied, unused vocabulary
projection is frozen; tied token embeddings remain trainable.

## Gates and execution

`scripts/finetune_study.py` creates the sequential job manifest, selects learning
rates from development results only, resolves selected seed-42 test inputs,
and validates the final paired result table. All paths are fresh. Raw files
remain in the already verified downstream benchmark snapshot; tokenization
happens inside each training entry point using the shared HF dataset cache.

Before production: three full-checkpoint FA4/SDPA forward/backward comparisons,
then four-update distributed training and save/reload evaluation for all three
arms. Reload smoke uses 32 development examples, never test metrics.
Copy/check the project's Accelerate config and inspect `accelerate env` before
launch; reverify known burn identities, stop only those workers, wait/recheck
free GPUs. Reuse `train_then_burn` to restore communicating burns on success or
failure. An unknown GPU owner blocks launch.

Local tests cover unchanged forward values, both attention paths (CPU FA4
reference), task gradients, padding, balanced truncation/EOS, actual Trainer
saving/best-model reload, and the real fine-tune/train→test CLI. Real CUDA
validation and smoke status must be recorded separately; local tests do not
prove B200 execution.

## Interpretation

This asks whether the architectures adapt better under supervised training.
It does not replace zero-shot or equal-pretraining-time comparisons. Three
fine-tuning seeds still share one pretraining seed. Prior arm selection used
exploratory results; a small gain requires independent confirmation. This
protocol is not a tuned leaderboard reproduction or a multilingual claim.

## Local verification and prepared submission

Six new CPU tests passed (3.861 s), including the actual training→reload→test
entry point, strict saved-state restoration, dev-only selection and queue order.
29 existing evaluation/proxy regressions passed (269.620 s); the original
pretraining all-arm cutoff/save/resume test passed (11.175 s). A test caught
an initial method-reuse/super() issue; the shared save helper was factored out
and these checks passed after the correction.

Prepared job: `th2-tjx3-supervised-finetune-20261007-a01`; root
`/mnt/local/_outputs/deep-llms_th2/supervised-finetune-20261007-a01`.
58 sequential stages including numerical gates, distributed smoke/reload,
search, selections, confirmations, final tests and final validator. Execution
and actual runtime remain to be verified.

First submission `1d179b7` was blocked before execution by the runner's
outbound-pattern guard: it matched the protective `args.push_to_hub` condition.
No process termination or GPU work occurred. Replaced the rejection condition
with explicit unconditional `args.push_to_hub = False`; no upload operation
exists or is enabled. Retry uses a fresh `supervised-finetune-20261007-a02`
job/root/session and the same scientific protocol.

The second submission `67ee50a` was likewise blocked before execution because
the guard also matches an explicit False assignment. The final entry point
uses the same native default-disabled Trainer configuration as train.py;
there is no upload branch or upload-specific code. The generated, fixed job
configs do not enable publishing, external reporting is disabled, and HF
libraries run offline. Fresh retry: `supervised-finetune-20261007-a03`. Neither
blocked submission changed GPU ownership or started any training.

## Remote acceptance progress

Accepted launch `0765d2e`, actual root `supervised-finetune-20261007-a03`.
Environment/data/checkpoint preflight passed. Project Accelerate config copied
and byte-verified; accelerate env reports MULTI_GPU, 8 processes, BF16. Known
burn workers 314038–314045 stopped at 17:27:42 Singapore; after 30 seconds,
all eight GPUs had no compute processes at 17:28:12.

All three full-checkpoint FA4-vs-SDPA gradient comparisons passed: relative L2
A .0106580, P6 .0109333, P7-simple .0107938. Every task-connected parameter had
a finite gradient; proxy aggregate gradients were nonzero. Buffers unchanged.

Monitor `9cff5a2`, 17:34:45 Singapore: A and P6 four-update training on all eight
GPUs and strict reload evaluations passed. Saved and reloaded development
loss/accuracy agree exactly, as do weight hashes and example-order hashes.
P7-simple distributed smoke still in progress; production fitting is not yet
verified. Evidence: artifacts/finetune-monitor-20261007-a03/ (11 verified JSON
artifacts). Log SHA256 db1d665882848b1e224983b13d9966f4c28d21c3fa68c3e277869da22496d427.

## Production fitting verified

Monitor `3717b86` at **17:41:09 Singapore, 7 October**: all three numerical
gates plus all six eight-GPU training/reload smoke stages passed. Reloaded
development loss and accuracy match the saved-checkpoint evaluation exactly
for all three arms, as do weight hashes and document-order hashes.

First production run, A / PAWS-X / seed 42 / LR 1e-5, was running on all eight
GPUs with finite loss/gradient logs, past epoch 2 (last log epoch 2.655).
Development accuracy was .879 after epoch 1 and .8885 after epoch 2. These
are development measurements for selection, not final test results; the other
arms and three-seed comparison remain pending.

Evidence: artifacts/finetune-monitor-20261007-a05/ (13 source-hash-verified JSON
artifacts), temp/finetune-monitor-20261007-a05.log SHA256
`120bcfe128d742442e85656de486c0201d082998a91b63535a2bdac22f1a22e0`.
The accepted detached queue continues through all remaining stages. Commands
returned to #0 to prevent accidental resubmission; automatic communicating
burn restoration remains configured on success/failure. Do not relaunch.

## Fine-tuning review and progress — 7 October 2026

Read-only monitor `790694a` at **11:07:15 UTC (19:07:15 Singapore)**:
14 of 24 production fitting runs completed: all 12 LR-search runs and PAWS-X
A confirmations at seeds 43/44. PAWS-X P6 seed 43 is running on eight GPU
workers 328924–328931. All six development selectors chose LR 3e-5.
No final test stage has run yet; all 18 tests follow completion of all fitting.
Queue and supervisor report running, with no failed stages. Do not relaunch.

Re-reviewed the task head, task-connected proxy gradients, standard Trainer/DDP
loss reduction (GAS=1), padding isolation, matched data/seeds, development-only
selection, and strict checkpoint restoration. No correctness defect found.
Six focused CPU tests passed again (3.923 s), including real entry-point
train/save/reload/test. Existing B200 numerical gates and all eight-GPU smoke/
reload stages passed. This evidence supports the implementation; it is not a
proof of absence of every possible bug. P7-simple deliberately enables
end-to-end task gradients only in the downstream wrapper; no pretraining
auxiliary loss is used.

Evidence: `artifacts/finetune-monitor-20261007-a06/`, 33 source-hash-verified
JSON artifacts. Source log SHA256
`0c2bcfdbf167f033a921b3c5632bfff0d5ee6c61ac6a5dfb2562422c0d538049`.
Focused test log: `temp/finetune-review-tests-20261007-a01.log`.
The earlier 11-model, nine-task zero-shot evaluation is complete. The supervised
comparison is still pending; commands returned to #0 without stopping the
accepted detached queue or its automatic final burn handoff.

## Supervised study complete — 7 October 2026

Read-only monitor `57be0de` at 13:41:06 UTC verified all 58 queue stages passed:
24 full fitting runs, six development LR selections, 18 final tests, numerical
and distributed smoke/reload gates, and the final prediction validator.
Queue finished 12:37:41 UTC (20:37:41 Singapore), elapsed 3 h 9 min 30 s
including gates, preprocessing, fitting, selection and testing. All six
selectors chose LR 3e-5. Each test uses 2,000 PAWS-X or 5,010 English XNLI
examples. Local checks matched all 18 result receipts to the summary.

Test accuracy, percent, mean ± sample standard deviation across fine-tuning
seeds 42/43/44:

| Arm | PAWS-X | English XNLI |
|---|---:|---:|
| A | 91.850 ± 0.278 | 78.922 ± 0.111 |
| P6 | 90.300 ± 0.218 | 79.188 ± 0.579 |
| P7-simple | 90.917 ± 0.029 | 78.696 ± 0.306 |

A leads PAWS-X. P6's NLI mean gain is only 0.266 percentage points, smaller
than its across-seed standard deviation; this does not establish a reliable
improvement. P7-simple trails A on both task means. These are three fine-tuning
seeds from one pretraining seed, with exploratory arm selection.

Automatic communicating burns were verified at 12:38:52 UTC. The 13:41:06
snapshot matches the same workers 341142–341149 on all eight GPUs at 100%
utilization. No training remains queued. commands returned to #0.

64 source-hash-verified JSON artifacts:
`artifacts/finetune-monitor-20261007-a07/`.
Log `temp/finetune-monitor-20261007-a07.log`, SHA256
`6d663b9f48eee94c00bac799480c0fc298fd6c0820870b340458f465213b5de6`.
Per-example prediction arrays remain on B200; the successful remote summary
validator checked their shapes, finite values, label/order consistency, and
recomputed accuracy. Small result receipts and summary have been pulled locally.

## P6-iso supervised extension authorized — 7 October 2026

User clarified that the strongest P6 variant should also receive fine-tuning,
and confirmed proceeding with P6-iso. Earlier P6 was stronger on zero-shot
benchmarks; P6-iso had the best held-out LM loss. Run P6-iso only and reuse the
completed A/P6/P7-simple results. Same tasks, data, seeds, LR search and Trainer
recipe: 8 full fitting runs and 6 final tests, preceded by CUDA and eight-GPU
smoke/reload gates. Fresh root supervised-p6iso-20261007-a01. No new pretraining.

The classification wrapper enables end-to-end task gradients through P6-iso's
predictor, as for P7-simple. This is a downstream-only mode: forward values,
checkpoint keys, and default pretraining gradient isolation remain unchanged.
No auxiliary objective or normalization updates. Numerical, gradient and
save/reload tests must pass before production. Reverify/copy Accelerate config,
stop only verified authorized burns, require free GPUs, and retain automatic
communicating burn restoration after success/failure.

Prepared launch: `th2-tjx3-supervised-p6iso-20261007-a01`, fresh root
`/mnt/local/_outputs/deep-llms_th2/supervised-p6iso-20261007-a01`.
20 stages: 1 CUDA gate, 2 distributed smoke/reload stages, 4 LR search fits,
2 development selectors, 4 confirmation fits, 6 final tests, 1 validator.
The checkpoint SHA256 is pinned to the already evaluated P6-iso weights:
`6bd57582187722caa6e7ca1b504993c40eb9cbc89f086079ca7f8b61d7ded8cf`.
Preflight also compares all generated task configurations to the previous P6
study, permitting only arm/checkpoint and fresh output/selection paths to differ.
No production execution is claimed until remote gates return successfully.

Local verification before this extension: all 24 fine-tuning/anticipatory-proxy
tests passed in 130.098 s. Includes both attention paths, default isolation,
P6-iso full entry-point save/reload/test, recipe matching, pretraining resume,
custom autograd/compilation and two-rank DDP regressions. Log:
`temp/finetune-p6iso-local-tests.log`.

P6-iso launch `892fc80` accepted. Initial runner receipt verifies environment
versions and pip check, byte-identical copied Accelerate config with eight
processes/BF16, pinned checkpoint/recipe hashes, local benchmark data, and all
20 stage configs matching the previous study. Verified burn workers
341142–341149 were stopped. GPU clearance and CUDA/smoke gate receipts are
pending the subsequent monitor; initial launch output alone does not establish
production training. Source log: temp/p6iso-launch-a01.log, SHA256
`18ff6d5290d65a49934eb9b6e5fdfa368751e88e01643c3345cb7b708b803c88`.

## P6-iso supervised fine-tuning running — 7 October 2026

Launch `892fc80`, root
`/mnt/local/_outputs/deep-llms_th2/supervised-p6iso-20261007-a01`.
Monitor `001a821` at 14:22:39 UTC confirms CUDA numerical gate, eight-GPU
four-update smoke and strict reload all passed. Reload loss/accuracy, weight
hash and development-example-order hash match exactly. Production PAWS-X
seed 42 / LR 1e-5 is running on eight workers 343599–343606; finite logs
through epoch .3238. No final results yet. Queue: 8 fits, 6 final tests,
matched protocol; reuse completed A/P6/P7-simple comparisons.
Accelerate config copied/verified, environment/data/recipes checked. Only
verified burn workers 341142–341149 stopped; all eight GPUs verified free
at 14:18:06 UTC. Automatic communicating burns remain configured after either
success or failure. commands #0 leaves the detached queue running; do not relaunch.
Evidence: artifacts/p6iso-finetune-monitor-20261007-a02/ (7 verified JSON artifacts).
Full protocol: docs/SUPERVISED_FINETUNING_20261007.md.

CUDA gate: all 348 task-connected parameter tensors have finite gradients,
nonzero aggregate predictor gradients, buffers unchanged. FA4-vs-SDPA
relative L2: gradients .0096595060, logits .0047431584.
Source monitor log SHA256:
`25afdd2b2c7fc5e53bfb4594c5e7e1682d83beaabd6cbcee57deb5bb3b568724`.

## P6-iso supervised evaluation complete — 7 October 2026

Monitor `6ef6e1d`, 23:41:08 Singapore: all 20 stages passed, including 8 full
fits and 6 final tests. Queue finished 23:21:40 Singapore (63 min 34 s).
Mean test accuracy ± sample SD across fine-tuning seeds 42/43/44: PAWS-X
90.717 ± .751%, English XNLI 78.955 ± .445%. Compared with A: -1.133 pp
PAWS-X, +.033 pp NLI; no convincing downstream gain. Both LR selections 3e-5.
Data/tokenizer and example-order hashes match A's completed evaluation.
Automatic communicating burns verified 23:22:46; same workers 351289–351296
active at 100% on all eight GPUs in current snapshot. No training queued.
Evidence: artifacts/p6iso-finetune-monitor-20261007-a03/ (26 verified JSON
artifacts). Full report: docs/SUPERVISED_FINETUNING_20261007.md.
commands #0. Do not relaunch without a new request.

Final comparison, test accuracy percent (mean ± sample SD, three fine-tuning
seeds; one pretraining seed):

| Arm | PAWS-X | English XNLI |
|---|---:|---:|
| A | 91.850 ± .278 | 78.922 ± .111 |
| P6 | 90.300 ± .218 | 79.188 ± .579 |
| P7-simple | 90.917 ± .029 | 78.696 ± .306 |
| P6-iso | 90.717 ± .751 | 78.955 ± .445 |

P6-iso seed accuracies 42/43/44: PAWS-X 91.15/89.85/91.15%;
XNLI 78.4431/79.2415/79.1816%. These results do not show a convincing
supervised downstream advantage despite its best LM validation loss.

Local verification checked all expected counts, summary against six result
receipts, matching data/tokenizer/example-order hashes versus A, and matching
current GPU worker identities to the successful burn receipt. The remote
summary validator also recomputed accuracy from finite per-example prediction
arrays and checked shapes/label order; those arrays remain on B200.
Source log: temp/p6iso-finetune-monitor-20261007-a03.log, SHA256
`8b1668ddcb0d559e71706f0c12d5da54a4e6c094b013314f80b4168766c3fbee`.
