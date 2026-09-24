# Next stage: distilling the jointly trained Deep teacher

The user authorized this stage on B200, including safely stopping the current
GPU burn, vLLM servers and other verified GPU workloads. Reinspect live process
identities immediately before stopping; no name-pattern or arbitrary group kill.
Every GPU experiment uses all eight GPUs, sequentially. Do not alter the completed
joint-v2 results, checkpoint files, or immutable source snapshot.

## Question and necessary baseline

Can a shallow-only branch recover the useful **block-4 correction** made by the
trained Deep teacher, without computing its extra clean block-20 source pass?
The target is the teacher adapter's delta at block 4, not the full block-20 state
and not a subtraction of states from different layers or trained models.

For each seed, load its exact final joint-v2 Deep checkpoint. First reproduce its
saved 2M-token dev evaluation, then evaluate the **same backbone with feedback
disabled**. Freeze that backbone and its teacher adapter. The earlier trained
Base/Shallow models have different backbone weights and cannot replace this
same-checkpoint baseline or the new matched student control.

Stop before student training if teacher feedback does not improve this baseline
by at least 0.0005 nats/target-token with a paired 95% CI entirely below zero.
This gate checks whether there is a useful correction left to distill.

## Fixed student experiment: distill-v1

Four scientific training runs: LM-only shallow student and PCC student for each
of the two existing seeds. Each pair shares byte-identical fresh student weights,
frozen parent checkpoint, data order, optimizer, update budget and evaluation.
The teacher and backbone do not receive any gradient or optimizer update.

- Pair s=4,d=20; student correction attention is 2 heads × 128, strict-past,
  shallow queries/keys/values only. Zero output projection reproduces the
  feedback-disabled backbone exactly before optimization.
- PCC uses ordinary LM loss plus teacher-RMS-normalized SmoothL1 correction
  loss, lambda=1, beta=1. LM-only uses exactly the same architecture and LM loss.
- Targets are detached and recomputed from that seed's frozen trained teacher.
  No cross-checkpoint target mixing, stale cache, or deep input to the student.
- Calibrate sigma_delta once per seed on its first 1,000,000 training input
  tokens, including the masked final partial context. Reject invalid/zero scale.
- 6144 updates, 32768 global input tokens/update, 201326592 inputs per arm;
  microbatch 1, two accumulation steps per rank on eight GPUs. Total student
  exposure is 805306368 tokens over four runs. Reuse the existing frozen training
  pool and seed permutations; do not resample or export raw text.
- AdamW lr=3e-4, betas=(.9,.95), eps=1e-8, matrix decay=.01 (gate vector/norms/
  biases excluded), 307 warmup updates, cosine to 3e-5, clip=1. Global target
  normalization for LM and target×hidden-size normalization for alignment.
- Same 262144-token monitor every 256 updates; 2M-token final dev per arm.
  Adapter-only checkpoints refer to the SHA256 of the full frozen parent and
  retain optimizer/history/per-rank RNG for strict resume.

The 201M budget is a separately versioned extension of the original frozen
probe's 20M student schedule, matching the longer joint-stage budget requested
by the user. All four budgets are fixed before seeing student outcomes.

## Gates and limits

Require PCC to beat its newly trained LM-only control and its feedback-disabled
backbone in both seeds, with paired 2000-resample 95% CIs. Require recovery >=25%
where recovery=(NLL_off−NLL_PCC)/(NLL_off−NLL_teacher). Also require improvement
against the prior trained Base as a practical quality floor, while recognizing
that it does not match the additional student-stage training exposure.

A positive result recommends the conditional target-semantics control; it does
not launch that training automatically or unlock a held-out test. The current
2M dev set has already guided research, so this remains exploratory. A negative
result stops; no automatic budget extension. Measure student-only evaluation
separately from training (which includes teacher-target generation). Full-context
one-pass scoring is tested; cached autoregressive decoding/speedups are not yet
claimed.

## Launch sequence and validation

1. Local tiny-model and real two-rank CPU tests: exact no-op, teacher delta
   reproduction, strict causality, no teacher/backbone gradients, real parameter
   updates, accumulated gradients, uneven eval, scale and exact resume.
2. Reinspect and reclaim user-authorized GPU workloads using pinned process
   identities. Hold the supported burn-disable marker only for this queue.
3. Two eight-GPU teacher audits/calibrations, both required to pass.
4. Two-update eight-GPU LM/PCC capacity checks, fresh-reducer next-update resume.
5. Four fresh scientific runs via run_experiments.py, then a CPU paired report.
   Capacity adapters never initialize scientific runs. Any operational failure
   stops the queue. Restore prior controller guard policy when it exits.

## Implementation and current launch status

Implemented in `pcc/distill_model.py`, `distill_training.py`, `distill.py`, and
`distill_report.py`; the earlier joint/frozen-probe code paths are preserved.
Local regression: 124 tests passed in 144.467 s; separate reclaim identity test
passed. Focused nine student/DDP/report tests passed in 23.517 s.

The first read-only B200 readiness job (commit 1d2d4b5) failed at the controller
before execution because `thiennh-p6-tpbw` had no Running worker pod. No workloads
were stopped and no student training began. Await runner restoration. Do not
repeat an executable submission merely to refresh its log.

After fresh CPU/ownership verification on the restored node, the concrete entry
point is `scripts/with_gpu_guard_disabled.py --owner <run-name> -- bash
scripts/launch_distill_b200.sh <fresh-root> <completed-joint-root> <inputs>
<fresh-ownership-json> <passing-cpu-readiness-json>`. Use system Python for the
lease/reclaim helpers (pidfd support) and pcc_joint for model code. The launch
script verifies readiness matches its exact pcc source before any signal, then
runs the two audits, two capacity/resume checks, four students and final report.
The supported guard marker is restored when the child command exits.
