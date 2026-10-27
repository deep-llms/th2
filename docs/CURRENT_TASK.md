# Current task

## Active: final Q/K/V-only recovery test, 27 October 2026 SGT

Project idea: [PROJECT_IDEA.md](PROJECT_IDEA.md). Overnight scope:
[KV_RECOVERY_OVERNIGHT_20261027.md](KV_RECOVERY_OVERNIGHT_20261027.md).
Report times in Singapore time (UTC+8). Workspace /disk/thuat/efficient_llm.

### Scheduled tick 3: placement complete; final bounded experiment prepared

The delayed read-only health 4dc992d was retrieved, then independent final health
60dcf16 verified the full queue and burn handoff at **09:11:42 SGT**. All three
placement arms completed 08:29:36 SGT; supervisor restored/verified eight-rank
communicating burns at 08:30:46 SGT. No overlapping workload or restart occurred.

BF16 PPL: matched dense A **19.2416**, uniform25 **20.4610** (+6.34%), upper25
**20.6189** (+7.16%), full50 **22.6557** (+17.74%). All fail the upper-95%-CI
<= log(1.05) quality screen. Uniform25 outperforms upper25 at equal 75% cache
retention. All actual sample-order/coverage, parent and reload checks pass.
[Placement results](KV_PLACEMENT_RESULTS_20261027.md), SHA-verified archive:
`docs/results/kv_placement_20261027/`. Raw log `temp/kv_placement_final01.log`,
SHA256 `1906ef653d0a038811026afbaeea5e9539b4e2dcdc9563a51d8723507283fad3`.
The previous B continuation (22.3924 PPL) has different starting weights and is
not a matched placement control. Its report is KV_RECOVERY_RESULTS_20261027.md.

One final controlled training job is prepared: uniform25 from the SAME original
dense A-512, updating affected Q/K/V projections only. Same 2048 steps, 134.22M
tokens, actual data order, teacher, loss, LR/schedule and runtime. Freeze all
other weights and require exact before/after byte identity. Compare completed
all-parameter uniform25 and dense A. This tests a focused optimization question;
restricted tuning and partial sharing are already known, not novelty claims.
See [KV_QKV_RECOVERY_20261027.md](KV_QKV_RECOVERY_20261027.md).

Local verification: 6 CPU tests including two-rank training, exact frozen-weight
checks and matched summary validation; 58 infrastructure tests. All pass.
The unchanged architecture's real GPU numerical gate already passed. New
8-step/8-GPU smoke must pass before the one real fit. Source manifest checked.
No environment, driver, dataset, model, core attention or loss changes.

This commit submits the fourth and FINAL additional training job beyond A/B:
`/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-qkv-20261027-a01`,
session `efficient_kv_qkv_20261027_a01`, study `kv_qkv`.
**3 completed + 1 submitted = 4/4 cap. No further training authorized.**
Submission is not launch/completion proof; retrieve receipt and smoke status.
If the quality screen fails, archive and stop this branch. If it passes, only
bounded actual cache/latency and longer-context evaluation remains justified.
No broad pruning sweep. Preserve every prior output/checkpoint.

QKV predecessor is completed placement a01, source burn path
`/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-placement-20261027-a01/source/resources/llm_pretrain_burn.py`.
At 09:11:42 SGT workers 618321–618328, launcher 618252; collective progress and
released guard verified. These are historical identities: supervisor must freshly
verify exact PID/start identities, hold guard, stop only known burn workers,
wait/recheck all eight GPUs free, then run. It copies resources/accelerate_b200.yaml
to actual HF cache and checks accelerate env; restores/verifies communicating
burns on completion/failure. Never signal unknown processes or groups.

commands.sh is a NEW QKV launch request replacing acknowledged final health.
Never repush it unchanged. Restore #0 after acknowledgement, using read-only
health with a unique identifier if fresh supervisor/progress evidence is needed.
Controller/AWS/credentials/connectivity failure freezes remote writes.

Existing dev timer only: PID 324788, tmux efficient_kv_recovery_watch_20261027,
state temp/kv_recovery_watch_20261027. Tick 3 delivered 09:08:46 SGT; next
**09:53:46 SGT**, deadline **18:53:46 SGT**. Do not create another agent, session
or timer. End this turn when the job is healthy; do not wait for long completion.
Touch STOP only after final artifacts and burn handoff are verified and the
research branch is concluded, or user cancellation/deadline. A delayed fetch is
not evidence of failure and not a reason to stop the timer.

Recovery a02/a1847c0 completed 07:39:16 SGT, archived with exact data and reload
checks and communicating burns. Original pilot a03 contains dense A-512 parent.
Placement bca23aa completed as above. All source copies on node omit .git;
never use git rev-parse in a node shell header. Source identity uses SHA manifests.

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

Latest verified burn ownership, **09:11:42 SGT**: workers 618321–618328,
launcher 618252; source placement a01/source/resources/llm_pretrain_burn.py.
Guard released and collective progress verified. QKV supervisor must
reinspect exact identities before stopping; these PIDs are historical evidence,
not permission to signal without rechecking. Guard/burn rules in GPU_SAFETY.md.

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
