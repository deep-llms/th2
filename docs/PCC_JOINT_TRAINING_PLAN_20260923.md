# Proposed joint-training experiment — 2026-09-23

Status: implemented and reviewed after the user's follow-up. Bounded local
correctness/capacity checks passed; the six scientific runs remain unlaunched.
See `PCC_JOINT_READINESS_20260923.md` and `PCC_JOINT_TRAINING.md` for evidence
and executable commands. The settings below define the new experiment.
This is a new experiment version; the completed frozen screen remains negative.
It is not an extension that changes that screen's eligibility rule.

## Recommendation and question

Keep all 28 layers of the pinned pretrained Qwen3-0.6B-Base and train the whole
backbone together with the added branch. Start from pretrained weights, not
random initialization. Use local A100s and `train_env`; B200 stays untouched.

Question: does allowing the backbone and feedback branch to adapt together
produce a useful deep-source advantage over equally trained shallow attention?
This experiment tests joint adaptation plus a larger budget; it does not isolate
which of those two changes explains any difference from the earlier screen.

Removing pretrained layers changes both the initial model and the depth of the
representations being compared. Recovery from that change would confound the
feedback question. Reduced-depth models are suitable for software tests, or a
separately designed small architecture study. They are not the primary pilot.

## Proposed fixed experiment

Three arms, each initialized from an independent copy of the same checkpoint:

1. Base: ordinary language-model continued training, all backbone weights trainable.
2. Shallow: same backbone plus the existing strict-past shallow-source branch,
   all backbone and branch weights trainable.
3. Deep: same backbone plus the existing strict-past deep-source branch,
   all backbone and branch weights trainable.

Use one fixed pair `(s=4,d=20)` for this bounded follow-up. It was not eligible
in the previous screen; choosing it here is exploratory and does not claim a
successful selection. It had the smallest deep-versus-shallow deficit in that
screen. No additional pair search is part of this plan.

Proposed budget: 1,536 updates x 32,768 non-padding input tokens = 50,331,648
input tokens per arm, context 2048. Two paired run seeds yield six runs and
301,989,888 total training input tokens across arms. These counts exclude extra
forward passes and validation; record actual GPU time and throughput as well.
All arms finish the same update count. No arm-specific loss-based early stopping.

Paired runs: (adapter seed 2901, data order 20260922), then (adapter seed 3901,
data order 20260923). Each run uses the same fixed context pool; Base, Shallow,
and Deep receive the same order within each seed. Seed changes are prescribed
before execution, not chosen after results.

Proposed optimization: AdamW beta=(0.9,0.95), epsilon=1e-8, clip global grad norm
at 1; backbone LR 1e-5, branch LR 3e-4; 77-update linear warmup and cosine decay
to 10% of peak. Backbone matrix decay .01; backbone norm/bias decay zero.
Branch decay follows the existing matrix-versus-gate/norm rule. Both parameter
groups use fp32 master parameters/moments and bf16 autocast. Preserve tied
embedding/head weights. These are initial design choices, not established optima.

## Data and evaluation

Use the pinned, already downloaded user-selected CulturaX English parquet.
Keep the existing validation document rows [20000,30000) excluded from training.
The current local packed training source has only 15,908,864 input tokens, so
it cannot supply this budget without repetition. Before any launch, extend the
fixed training source with unused document rows after 30000 from the same local
parquet, then freeze the context pool and exact counts. Use existing shared
packing/loading; do not introduce a user-facing text-export stage or modify
`prepare_data.py`. Do not silently recycle short inputs or claim repeated tokens
as additional unique data. All final inputs must be ready before training.

Record fixed-set validation at step 0 and every 128 updates using the same
262,144-input-token monitoring slice. Evaluate all arms on the same 2M validation
tokens at final step 1,536. Keep per-context summed losses/counts. Use the final
step for the primary comparison, not each arm's individually best checkpoint.
Save learning curves for raw training loss, fixed validation loss, and contrasts.

The local validation set has already been inspected in the frozen experiment;
this remains exploratory evidence. No confirmatory test claim or automatic
B200 scaling follows. Report paired 95% bootstrap CIs (2,000 resamples per seed,
seed 20260922), effect sizes, both training seeds separately, and their spread.
Context-bootstrap uncertainty is not a substitute for training-seed uncertainty.

Proposed next-stage recommendation requires Deep to beat both trained Base and
trained Shallow at the final step in both seeds, with upper paired 95% CIs below
zero and at least .0005 nats/token improvement over trained Shallow in each seed.
This is a new practical gate, not a retroactive change to the frozen protocol.
Otherwise report negative/inconclusive as appropriate. A still-falling curve at
the budget limit is evidence of incomplete convergence, not permission to extend.
No automatic budget increase, new pair search, distillation run, or remote job.

## Required implementation before execution

The current `FrozenQwen` and correction adapter detach backbone states, and
`clean()` uses `no_grad`. Simply adding backbone parameters to the optimizer
would not implement joint training correctly. Add a separate trainable path:

- Base uses the normal differentiable full forward.
- Shallow injects its branch at s during one differentiable forward.
- Deep computes clean states through d, then reruns the corrected tail from s
  with shared current-checkpoint parameters. Keep both the shallow state and
  deep source differentiable; do not use the frozen path's detach operations.
  Optimize the final corrected LM loss. No teacher alignment or additional
  clean-pass loss in this stage; all arms optimize their final LM prediction.
- Strict-past visibility and post-block coordinates remain unchanged. The
  two-pass computation is an acyclic graph; this is not recurrent decoding or
  an exact reproduction of Full-bandwidth Transformer / WhiteMatter.
- Activation checkpointing must retain gradients through both passes and avoid
  capture-hook interference on backward recomputation. Chunk vocabulary loss
  to bound memory. Preserve zero-initialized branch outputs.
- Add fixed-set periodic validation and save full resumable checkpoints:
  backbone, branch, optimizer, schedule, update, RNG, sampler/data cursor,
  configuration, and input identity. Checkpoint after completed updates only.
- Verify nonzero gradients and actual updates in early and late backbone blocks,
  source-state gradient flow, causality, initial no-op equivalence, matched
  accumulation, tied weights, and save/resume continuation on small fixtures.
  Preserve existing frozen-mode tests and behavior.

## Bounded unattended execution after implementation

Use a sequential job manifest compatible with `run_experiments.py`, with explicit
dependencies, fresh outputs, status/failure markers, and a final report. A short
full-size capacity/throughput check on local A100 hardware establishes memory
and runtime; discard its weights before scientific runs. No training launched
during planning. Do not promise a runtime from the frozen screen's throughput.

Default scientific scheduling: one arm at a time on one verified-free local
A100, all six jobs queued automatically. This minimizes peak resource use and
avoids distributed-training complexity; it trades longer wall time for simpler
execution. Choose the same feasible microbatch for all arms, accumulate to the
fixed global token count, and use activation checkpointing. Do not silently
reduce depth/context/budget after an OOM. Stop and report if the capacity check
cannot run the agreed model even at microbatch 1.

Save periodic full state every 128 updates; retain latest and final full state
plus metrics. Implement and test explicit resume; do not retry indefinitely or
interpret a timeout as successful completion. Software failure stops the queue.
The final report states all completed budgets, failures, loss curves, paired
comparisons, and compute cost. This task introduces no B200 submission.

## Research references for the model-size decision

- [Layer pruning study](https://arxiv.org/abs/2403.17887): layer selection and
  recovery finetuning are part of the pruning procedure. Its results do not
  validate arbitrary layer removal from Qwen3-0.6B.
- [Full-bandwidth Transformer](https://arxiv.org/html/2608.08888v1): trains with
  feedback, permits starting from standard checkpoints, and allows gradients
  through passes. This motivates adaptation, without making this pilot a
  reproduction of that architecture or evidence that it will succeed.
