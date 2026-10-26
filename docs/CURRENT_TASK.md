# Current task

## Active: KV recovery and overnight continuation, 27 October 2026 SGT

User approved a matched recovery test and bounded autonomous follow-ups while
sleeping. Design: [KV_RECOVERY_OVERNIGHT_20261027.md](KV_RECOVERY_OVERNIGHT_20261027.md).
A/B will start from the old step-512 final weights; explicit new optimizer and
2,048-step schedule, 134.22M fresh tokens per arm. Reuse the old 256 validation
prefixes and exclude all old pilot documents from training. Same model, SDPA,
frozen teacher, native Trainer and 8-GPU recipe. Eight-step B smoke gates A/B.

Read-only preflight `d9ed973` passed at **06:44:15 SGT**: parent hashes verified,
21,710 GiB free, only the previously identified eight burn workers 612725–612732,
launcher 612656; guard released; collectives advancing. Receipt:
`temp/kv_recovery_preflight01.log`. Controller healthy. Reinspect before reclaim.
CPU data/score tests (4), model/2-rank training tests (8), recovery decision test (1),
and infrastructure checks (56, including 8 job tests) pass.
No environment or model architecture changes; GPU lifecycle unchanged.

Submitted `6399e48` at approximately 06:53 SGT; no launch receipt in Dropbox as
of the 06:58 SGT listing. Latest controller status remains the successful
preflight. Do not resubmit or infer failure from this delivery delay. Keep the
submitted command until acknowledgement, then restore #0. Training is not yet
verified running. Timer must check this pending handoff first.
Fresh output
`/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-recovery-20261027-a01`,
predecessor training `6y3o-efficient-kv-mlp-20261027-a03`, predecessor burn source
`6y3o-efficient-kv-mlp-inference-20261027-a01/source/resources/llm_pretrain_burn.py`.
Session `efficient_kv_recovery_20261027_a01`. CPU data preparation happens before
reclaim; Accelerate cache copy/env, exact worker identities and free checks are
mandatory. Final artifacts: `supervised/run/{A,B}/result.json`, `summary.json`,
`complete.json`, `supervised/supervisor.json`, `burn-verified.json`.

Schedule this existing conversation every 45 minutes, at most 12 hours, with
`scripts/queue_watch.py`; state under ignored `temp/kv_recovery_watch_20261027/`.
Timer is live (PID 324788, tmux `efficient_kv_recovery_watch_20261027`). First tick
07:38:46 SGT; deadline 18:53:46 SGT on 27 October. App server accepted delivery
test message `01a17a6b-7b31-7c12-a366-27b05ce6cebf`; actual turn delivery remains
pending. Check `delivery_ack.json` for confirmation, not the enqueue receipt.
Do not create another timer or agent if one exists. At most four extra training
jobs (<=4096 steps each) beyond A/B, justified by results. No broad unmotivated
sweep. Upper 95% paired-document CI <=log(1.05) gates pruning after recovery.
If the sharing gap remains large, inspect a focused conversion hypothesis instead.
Respect latest user steering, controller failure rules and the safe burn handoff.
Follow-up training jobs submitted beyond A/B: **0 of 4**.
End long-wait turns; the timer queues the next turn. Stop scheduling once useful
work/results are complete or the cap expires. Record actual delivery separately
from successful enqueue. No remote follow-up is authorized by stale PID state.


Workspace `/disk/thuat/efficient_llm`, independent efficient-LLM architecture
project. Report times in Singapore time (SGT, UTC+8). Objective:
[PROJECT_IDEA.md](PROJECT_IDEA.md); original pilot design:
[KV_MLP_PILOT_20261027.md](KV_MLP_PILOT_20261027.md).

## Completed initial pilot (historical)

User approved a bounded pretrained KV-sharing × MLP-pruning interaction pilot.
All six fresh 512-step arms completed successfully on original 28-layer
Qwen3-0.6B-Base. Total 201,326,592 recovery tokens. Native HF Trainer/Accelerate,
native SDPA, identical actual sample order and teacher-distillation recipe.
Final training-validation PPL: A 20.03790, B 26.22461, C 21.67554, D 28.69731,
E 30.77849, F 32.49516. Recalibrating after sharing did not help this recipe.
This is a 256-document × 1024-target prefix probe, not full-corpus/downstream PPL.

Training source `7abbd37`, run `6y3o-efficient-kv-mlp-20261027-a03` completed
**05:26:25 SGT, 27 October**. Burns restored/verified 05:27:36; independent
health `3ba782c` at 05:28:28 confirmed advancing eight-rank burns and released
guard. Compact result bundle exists under this run's `result_export/`.
Its manifest SHA256: `57d09d8d87d4794401ac3e2c317850f6f77132ea45a381cbff83537f70048d02`.

Evaluation-only precision check `c140449` completed **05:39:00 SGT** in 34.67s:
`6y3o-efficient-kv-mlp-inference-20261027-a01`. BF16-weight PPL A/B/C/D/E/F:
20.03850 / 26.22722 / 21.67890 / 28.70302 / 30.78658 / 32.50281, matching the
profile precision with unchanged ranking. Same probe and trained model code
verified. Accelerate copied/verified, eight-GPU BF16 env confirmed.
Burns restored 05:40:11; independent health `381f1da` at **05:42:22 SGT** verified
advancing burns and guard released. No environment/driver changes.

**Archival complete:** exact-file export `8809832` returned 29 files, all
SHA256-verified locally against node receipts (folder export `59e30f9` failed
without changing results). Full local archive `temp/kv_mlp_results_20261027/`;
small durable artifacts `docs/results/kv_mlp_20261027/`.
[Final report](KV_MLP_RESULTS_20261027.md). Model/optimizer checkpoints remain
on temporary B200 storage; they were not pulled through the small-file channel.

Step-512 interaction D−B−C+A is +0.011546 nats [0.006544,0.018057]. E−D is
+0.070013; F−E +0.054275. Post-recovery single-layer pruning sensitivity
correlation A/B: Pearson .985, Spearman .922. B8/8k eager decode: A 18.390 ms,
D 15.821 ms (paired speedup 1.162×), but D's inference PPL is 43.2% higher.
Shared arms store exactly half the KV. No useful quality/cost joint benefit;
retain as a negative pilot rather than scaling this recipe. One short recovery,
model and seed; no 8k-quality or downstream claim. commands.sh #0.
**Historical pilot complete. Superseded by the recovery authorization above.**

## Verification and historical failures

Eight CPU model/training tests and 55 infrastructure checks pass. Actual-weight
GPU gate and eight-GPU smoke passed. Every arm passes exact one-pass data
coverage, matched per-rank order, frozen teacher and save/reload checks (max
checked NLL difference zero). Checkpoints require `load_compressed`, not vanilla
AutoModel; consumer KV retains gradients; decoder checkpointing is rejected.

Attempts a01/a02 stopped in the gate before training; outputs preserved. Corrected
reference QK precision and separated full-model same-kernel BF16 checks from
fixed-input cross-kernel checks. FP32 full explicit-math output/gradient errors
4.53e-5/9.91e-5; same-SDPA independent BF16 decoder output error zero; fixed-input
BF16 output/gradient maxima .001873/.012473. Full-model explicit-math BF16 errors
.399/.681 remain documented; cross-kernel interchangeability is **not accepted**.
See `docs/results/kv_mlp_20261027/gpu_gate.json` and the design document.

## Runner and latest ownership

Remote `git@github-share:deep-llms/th2.git`, main; this worktree serves development
and execution. Node `thiennh-p6-6y3o-worker-0`, eight B200s. Code
`/mnt/local/deep-llms_th2`; all outputs under `/mnt/local/_outputs/deep-llms_th2`.
Do not repeat migration/cleanup; never repush unchanged executable commands.sh.

Python `/mnt/local/conda-py311/envs/train/bin/python3.11`: torch 2.14.1+cu130,
transformers 5.9.0, accelerate 1.13.0, datasets 4.8.5. Model
`/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6`; data
`/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513`.
Contract `resources/cx_english_20261020.json`; config `resources/accelerate_b200.yaml`.
No GPU-node outbound network; controller #d/#i inputs and #2 exports only.
Dropbox label 6Y3O; credentials/map under ignored DOT/temp. Never print contents.

Latest burn ownership, **05:42:22 SGT**: workers 612725–612732, launcher 612656;
source `6y3o-efficient-kv-mlp-inference-20261027-a01/source/resources/llm_pretrain_burn.py`.
All eight GPUs owned by these known burns; guard released. This supersedes all
previous PID records. Reinspect exact PID/start identities before a later stop.
Guard/burn hashes and rules in GPU_SAFETY.md.

## Earlier evidence

[WINDOW_COMPACT_RESULTS_20261027.md](WINDOW_COMPACT_RESULTS_20261027.md),
[SELECTIVE_DECODE_RESULTS_20261026.md](SELECTIVE_DECODE_RESULTS_20261026.md),
[CONTEXT_BUDGET_RESULTS_20261026.md](CONTEXT_BUDGET_RESULTS_20261026.md),
[ROUTER_PILOT_RESULTS_20261026.md](ROUTER_PILOT_RESULTS_20261026.md), and
[JOINT_GROUP_RESULTS_20261026.md](JOINT_GROUP_RESULTS_20261026.md).

Prior learned group/context routing has not demonstrated useful whole-model
speed over the fastest dense control or cache-capacity savings. Compact kernel
is opt-in; native SDPA default. Removing sinks from the frozen half-local policy
severely worsens PPL. Do not restart those queues automatically. Raw archives
remain under ignored temp; durable reports/results under docs/. Large training
checkpoints remain on temporary B200 storage; small metrics are backed up.
