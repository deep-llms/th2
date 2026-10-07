# Supervised downstream study: A, P6, P7-simple

User authorized these three arms (P6, **not P6-iso**), starting from their
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
