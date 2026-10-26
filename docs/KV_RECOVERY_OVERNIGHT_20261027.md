# KV-sharing recovery and bounded overnight continuation

Authorized 27 October 2026 SGT: judge the proposed A/B recovery, implement/test/run
it, then keep investigating useful efficiency directions while the user sleeps.
Use a real scheduled wake-up rather than keeping an agent turn polling a long run.
This supersedes the previous “no further training queued” state.

## Judgment

The matched recovery pair is justified as a budget diagnostic. B's PPL continued
improving (31.43 at 128 steps, 26.22 at 512), but the 31% gap to A is substantial.
It is not justified to prune B further before sharing becomes usable. The old
E/F calibration was performed while sharing was badly disrupted; recovering
sharing first removes that confound. Even success would establish a conversion
baseline, not a novel research contribution. The original interaction signal was
small, and post-recovery layer sensitivity ranks were highly similar (.922).

[GQA](https://aclanthology.org/2023.emnlp-main.298/) demonstrates conversion followed
by uptraining using 5% of original pretraining compute, but it shares heads within
a layer, so it does not guarantee successful cross-layer conversion here.
[Original CLA](https://arxiv.org/abs/2405.12981) reports models trained from scratch;
its result is not evidence that our abrupt pretrained conversion should recover
within this small token budget. These are prior baselines, not our novelty.

## First stage: approved matched pair

- Original pretrained 28-layer Qwen3-0.6B-Base; existing pilot A/B final weights.
- New explicit fine-tuning phase: fresh AdamW optimizer and cosine schedule,
  peak LR 5e-5, 102 warmup steps (5%), weight decay .01, clipping 1, seed 42.
  The completed old 512-step schedule is not silently extended/resumed.
- 2,048 extra steps per arm, all eight B200 GPUs, batch 8/GPU, accumulation 1,
  length 1024: 134,217,728 fresh tokens per arm, 268,435,456 total.
- Same frozen original teacher and loss 0.5 CE + 0.5 KL(T||S), T=1. All student
  weights train. Architecture, SDPA backend and precision unchanged.
- English nht10/cx_sampled, pinned qwen3_0.6b_base_en_30B subset. Choose 131,072
  fresh document prefixes, seed 9042, scanning at most the first 32 pinned shards.
  Exclude all previous pilot training/calibration/validation text and token hashes.
  Reuse the exact old 256 validation prefixes. This is development validation,
  not a new independent confirmation set. Never select training data by loss.
- Verify parent hashes/architecture and require initial per-document validation
  NLL to match the parent result within 1e-5. Exact sample-order equality and
  one-pass coverage across A/B; frozen teacher; checkpoint reload tests.
- Validate and save every 256 steps; retain all checkpoints. Score final BF16
  inference weights, matching the existing profiler's precision.
- Two-rank CPU continuation/data tests; eight-step eight-GPU smoke from B before
  full A then B. Smoke outputs separate; smoke never updates source checkpoints.
- Existing safe GPU supervisor unchanged: CPU preparation with burns intact,
  copy/check Accelerate + accelerate env, verify exact burn identity, reclaim
  known workers, wait/recheck free GPUs, run jobs, restore communicating burns
  and release guard on success/failure. No env/driver installation or changes.

## Decisions after completion

The screening gate is the upper 95% paired-document CI for B−A NLL <= log(1.05),
using BF16 inference scores. This is a practical conservative gate, not a downstream
quality guarantee; reused validation and one seed limit inference.

1. If the gate passes, test MLP pruning calibrated on recovered A/B with matched
   recovery budgets and fixed data. Include a control using dense-derived pruning
   on B to distinguish a benefit of recalibration from recovery alone. Validate
   2k/8k quality and actual cache/speed before presenting a useful frontier.
2. If the gap remains large, do not launch another broad pruning sweep. Inspect
   the learning curve and research/test at most one focused conversion hypothesis
   at a time (e.g. less destructive fixed sharing placement or gradual conversion).
   Account for existing cross-layer/head-sharing work before claiming novelty.
3. Throughput changes require numerical/gradient checks and whole-model timings.
   Do not spend on custom kernels until quality and a real runtime bottleneck
   justify them. Keep the validated backend as the control.

Bound follow-up authority conservatively: at most four additional training jobs
beyond this pair, each at most 4,096 steps with the same 65,536-token global batch,
and at most 12 hours of scheduled continuation. Every job must answer a specific
question with a matched control. These are caps, not quotas: stop an unpromising
branch, preserve results and leave verified burns. No model/data/env replacement,
large search, external publication or new agent without further instructions.

## Scheduled continuation

Use `scripts/queue_watch.py` on the dev machine to enqueue messages on this
existing conversation through `codex queue --remote unix://`. No separate agent,
new session, competing writer or API credentials. Official documentation describes
[queued prompts waiting for the active turn](https://developers.openai.com/blog/mastering-codex-remote-for-engineering).
The previously tested local CLI supports this queue command.

Interval 45 minutes; start after launch, stop after 12 hours or local STOP file.
State/prompt/logs under ignored `temp/kv_recovery_watch_20261027/`; tmux
`efficient_kv_recovery_watch_20261027`. Check queue receipt and live timer; distinguish
successful enqueue from actual later turn delivery. It depends on the dev machine,
app server/authentication and model availability; do not promise phone notifications.

Each wake: read CURRENT_TASK and latest user messages, inspect one bounded stage,
retrieve fresh safe #2/#1 evidence if required, verify exit/artifacts and burn
handoff, update durable results and decide the next meaningful experiment. Never
repeat an unchanged executable commands.sh or start overlapping work. No polling
until the next tick: end the turn once a long job is healthy. On user cancellation,
stop scheduling. On controller/AWS/credentials failure, freeze remote writes and
report; do not repair infrastructure through GPU process kills or relaunches.
Stop the watcher when conclusions/artifacts/burn handoff are complete or the cap
is reached. Restore commands.sh #0 after acknowledged actions.

## Initial submission receipt

CPU checks: 4 data/score, 8 model/distributed recovery, 1 decision-gate test, and
56 infrastructure tests passed. Read-only B200 preflight passed 06:44 SGT.
Attempt6399e48 failed before Python/GPU work because its shell header requested
Git metadata from the node's source-only copy. Corrected a1847c0/a02 reached
Python and verified both checkpoint hashes at 07:00 SGT. No training ran in a01.
At the first a02 snapshot, probe preparation/smoke/training are not yet confirmed
complete; see CURRENT_TASK for live state. Existing data/env/old results preserved.

The live dev timer starts checks at 07:38:46 SGT, repeats every45minutes and ends
18:53:46 SGT on27October. Its delivery test was accepted by the app server; actual
receipt is recorded separately in ignored temp/kv_recovery_watch_20261027.

## Focused prior-work check during first monitoring tick

[SwiftKV, EMNLP 2025, sections 3.3–3.4 and 5.3](https://aclanthology.org/2025.emnlp-main.1306.pdf)
combines later-layer KV sharing with another transformation that projects later
KV from an earlier hidden state. Its distillation updates affected Q/K/V
projections and freezes the remaining weights. Its restricted-update ablation
beats updating all later-layer parameters. Those experiments use substantially
larger instruct models and instruction data, so they do not establish the right
recipe for our 0.6B base model and English continued pretraining.

Implication for our study: unrestricted sharing everywhere and full-model
updates are choices to test, not requirements of sharing. If the recovery gap
remains large, a matched study of less destructive sharing placement or narrower
parameter updates is better motivated than pruning the already damaged model.
Neither change is novel by itself. No follow-up is selected or launched from
this literature check alone; the current recovery curves come first.
