# Current task

## Running: 1.36B-token dense/uniform25 pair, 27 October 2026 SGT

Latest user authorization: "ok, training even longer is ok." New matched pair
supersedes the completed overnight cap for these TWO jobs only. Design:
[KV_LONG_RECOVERY_20261027.md](KV_LONG_RECOVERY_20261027.md), objective
[PROJECT_IDEA.md](PROJECT_IDEA.md). No further sweep or timer authorized/created.

**Latest verified status22:44:49 SGT:** dense A training on all8B200, validation
at step12,288/20,752 (59.2%), PPL18.06459 vs initial20.03790. Uniform25 B remains
queued; neither full fit is complete. Eight known training workers626614–626621,
~49.72GiB each, guard held by native supervisor. No errors in fetched logs.
Do not stop/restart or submit duplicate work. Old overnight timer remains stopped.

Recent pace0.2391 seconds/update projects A finishing~23:18 SGT27October and the
pair~00:40–01:00 SGT28October, including remaining B and handoff. Preparation
added~2h48 before GPU work; the earlier~3h estimate was training time only.

Data completed:1,328,128 unique1024-target prefixes, exact old131,072-prefix
extension and frozen256-doc validation, no original-parent overlap. SHA contract
checks passed over126 consumed source shards. Probe manifestSHA
f6328dca7e38da8a85a14e331d8a7247fc8a0723c6ab951b9e4289abb3ed9f75.
Accelerate config copied to actual cache and environment verified8GPU/BF16.
Known burn workers safely reclaimed; all8GPUs verified free21:52:38 SGT.
GPU uniform25 numerical gate and separate eight-step eight-GPU A/B smoke tests
passed; both smokes have identical parent hashes and actual per-rank sample order,
zero parent/reload NLL difference. Full-run order checks occur after completion.

Both full fits start from original dense A-512, not previous recovery final
weights.20,752 updates /1,360,003,072 recovery tokens each; fresh optimizer and
cosine schedule,1,038 warmup steps. Same teacher/loss/runtime/data; all parameters
trainable. Checkpoints retained every1,024 plus5,188/10,376/final. Native supervisor
runs A then B, validates comparison, and restores/verifies communicating burns
on success/failure. No environment, driver, architecture or loss changes.

Run `/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-long-20261027-a01`, session
`efficient_kv_long_20261027_a01`; launch1c968cf acknowledged19:04:22SGT.
Preparation session completed normally; external log is run-path plus`-prepare.log`.
Health3fbef5b receipt`temp/kv_long_health02.log`, SHA256
 a97cb215b1f930232047d121ed822cd7e1b2f79888e9bb3efc0dbc178b9f80d9.
12 small artifacts verified and archived under
`docs/results/kv_long_20261027/status_2244/`; live GPU inspection included.
Local33 model/data/job tests +42 infrastructure checks passed before launch;
extra long-data/handoff tests and two-rank intermediate-checkpoint test passed.
commands.sh restored#0 after acknowledged read-only health. Next status request
may reuse`temp/kv_long_health_command.sh` with a NEW job ID(a01/a02 already used).

## Completed: bounded KV-sharing recovery research, 27 October 2026 SGT

Project objective: [PROJECT_IDEA.md](PROJECT_IDEA.md). Authorized overnight scope:
[KV_RECOVERY_OVERNIGHT_20261027.md](KV_RECOVERY_OVERNIGHT_20261027.md).
Final report: [KV_RECOVERY_FINAL_RESULTS_20261027.md](KV_RECOVERY_FINAL_RESULTS_20261027.md).
Workspace /disk/thuat/efficient_llm. Report times in Singapore time (UTC+8).

All four additional training jobs in that earlier authorization are complete.
Final matched BF16 PPL: dense A 19.2416, uniform25 20.4610, upper25 20.6189,
full50 22.6557, uniform25 Q/K/V-only 23.0779. All sharing variants fail the
upper-95%-CI <= log(1.05) gate. Stop this short conversion/recovery recipe;
no pruning sweep or speed-only evaluation. One model/seed, reused 256-document
prefix probe: do not generalize this to all KV sharing or downstream accuracy.

### Final QKV job and evidence

Run `/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-qkv-20261027-a01`,
session `efficient_kv_qkv_20261027_a01`, launch3932ad4; final healthd1e92f3.
Receipt confirms Accelerate cache copy and eight-GPU BF16 verification. Smoke
8steps/8GPUs and full2048-step fit pass. Same original dense A-512 parent,
134,217,728 tokens, exact sample order and recipe as dense/all-parameter controls.
Frozen bytes unchanged; selected Q/K/V weights changed; parent/reload max NLL
difference zero. Local recomputed summary equals node summary exactly.

Queue completed **09:36:37.676 SGT**. Communicating burn handoff verified
**09:37:48.530 SGT**. Independent live health **09:55:46.983 SGT** confirms eight
known burn workers, guard released, advancing collectives through 1,250 cycles
and 1,387.94 GiB logical payload/rank. No environment/driver/model/data change.

13 exact exported artifacts SHA-verified, archive
`docs/results/kv_qkv_20261027/`; local `temp/kv_qkv_results_20261027/`.
Raw health `temp/kv_qkv_health01.log`, SHA256
`3ffd72cea3ebd77556ac10ea603a412dd10732fe3eaa9e98abe75e7f8db51e32`.
Large checkpoints remain on B200. Prior outputs and caches preserved.

Latest burn source is this QKV run's `source/resources/llm_pretrain_burn.py`,
launcher620091, workers620160–620167. These are historical identities, never a
substitute for fresh PID/start verification before any future authorized stop.
No active training or evaluation remains. commands.sh restored to **#0**.

Existing timer `temp/kv_recovery_watch_20261027/STOP` is set after verified final
results/handoff. No replacement timer/agent. Do not restart research from stale
queued wake prompts; completed cap is4/4. New authorization is recorded above.

### Earlier recovery stages

Recovery a02/a1847c0 complete07:39:16SGT: denseA19.2416, oldB22.3924 (+16.37%).
B starts from already-shared B-512 and is not a matched placement initialization.
Three placement jobs bca23aa complete08:29:36SGT from SAME denseA-512:
uniform25 +6.34%, upper25 +7.16%, full50 +17.74%. All artifacts/data/reload and
burn checks pass. Reports KV_RECOVERY_RESULTS_20261027.md and
KV_PLACEMENT_RESULTS_20261027.md; SHA archives under docs/results/.

Node source copy omits .git; source identity uses SHA manifests. Earliest recovery
a01 failed before Python/GPU because of a git metadata shell header; corrected
a02 preserved all outputs. Delayed Dropbox receipts are not run failures.

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

Latest verified burn ownership, **09:55:46 SGT**: workers620160–620167,
launcher620091; source QKV a01/source/resources/llm_pretrain_burn.py.
Guard released and collective progress verified. Reinspect exact identities
before any future authorized stop. Rules in GPU_SAFETY.md.

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
