# Current task

Working folder: `/disk/thuat/efficient_llm`. Separate efficient-LLM project;
normal layer order, no B-DOT or proxy requirement. Report times in Singapore time.

## Objective and scope

Design: [PROJECT_IDEA.md](PROJECT_IDEA.md),
[FIRST_ARCHITECTURE_HYPOTHESIS.md](FIRST_ARCHITECTURE_HYPOTHESIS.md).
Implemented first screen: [JOINT_GROUP_DIAGNOSTIC.md](JOINT_GROUP_DIAGNOSTIC.md).
The user authorized implementing and running it on the existing th2 B200 after
migration. No new environments, old-checkpoint cleanup or unrelated jobs.

Frozen original 28-layer Qwen3-0.6B-Base; retain 6/8 and 4/8 GQA groups at every
layer. Dense/static/raw-norm/projected-norm/joint comparisons. Dense execution
with masked outputs; actual modified-model trajectories. No speedup claim,
router fit or backbone training. A useful held-out LM advantage would justify
later matched-router fitting; do not expand training automatically.

64 calibration train documents, 128 validation documents, fixed seed 42,
2,048 next-token targets per long-document prefix. Exact row/token hashes and
consumed-shard checksums are frozen before evaluation. This is not full-pool PPL.

## Verified deployment and assets

Execution: `git@github-share:deep-llms/th2.git`, `main`, node
`thiennh-p6-6y3o-worker-0`; eight B200s. Current repository serves development
and execution: never repush unchanged executable commands.sh.
Code root `/mnt/local/deep-llms_th2`; output/data/models are separate external
roots. Dropbox credentials and 6Y3O map remain in ignored DOT/temp; do not print.

Migration verified at 09:05:45 SGT on 26 October 2026. All 56 initial source
hashes matched; old code archived with hashes and all burn workers preserved.
See `docs/results/efficient_llm_migration_20261026.json`. Do not repeat cleanup.
The old missing-path error was a missing receipt caused by a parser exception,
not a missing project directory. Fresh a04 verification resolved delivery.

Train Python `/mnt/local/conda-py311/envs/train/bin/python3.11`:
torch 2.14.1+cu130, Transformers 5.9.0, datasets 4.8.5, Accelerate 1.13.0.
Model `/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6`.
Dataset `/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513`.
The old model directory had tokenizer/config only. Controller d2ed6f0 downloaded
the original pinned safetensors; the actual weight hash and consumed shard hashes
passed preparation in launch 1f786de. No installed environment was changed.

## Current execution

Source/launch commit `1f786de`; run root:
`/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-joint-groups-20261026-a01`.
Independent supervisor session `efficient_joint_20261026_a01`, using immutable
`source/`. Local launch receipt `temp/th2_migration_20261026/launch01.log` confirms
PROBE_PREPARED and DIAGNOSTIC_SUBMITTED. Accelerate config was copied to its
actual cache and `accelerate env` confirms MULTI_GPU, 8 processes, BF16.
Read-only health `c1b710f` verified successful completion at 09:29:27 SGT,
26 October 2026. Both jobs exited zero. All eight communicating burns were
restored and verified at 09:30:38 SGT, still present at 09:32:02 SGT; guard released.
Workers 590625–590632, launcher 590556, source under this run’s immutable snapshot.
Export `b850d88` retrieved and verified locally: result/check hashes and counts,
probe-manifest hash, and source identity all match. Results and interpretation:
[JOINT_GROUP_RESULTS_20261026.md](JOINT_GROUP_RESULTS_20261026.md).

Sequence: verified burn-worker reclamation → all eight GPUs free → GPU correctness
check → eight-rank study → free check → communicating burn restored and verified.
Reused the earlier checked supervisor unchanged. On job failure it also restores
burns after cleaning only its own descendants; unknown ownership fails closed.

Expected outputs: `supervised/run/gpu-check.json`, `diagnostic.json`, `complete.json`,
`supervised/supervisor.json`, `supervised/burn-verified.json`.
All expected artifacts and final ownership verified. Reports are archived under
`docs/results/joint_groups_20261026/`; commands.sh restored to #0. No follow-up
training is queued. Next scientific step, if pursued: matched causal routers at
6/8 groups; neither efficiency nor dense-quality parity has been established.

## Local validation

59 tests passed: 48 helper/supervisor tests, 11 mechanism/scoring tests. Includes
GQA projection algebra, bias, causal/cache consistency, keep-all identity, EOS,
corrupted shards, shifted loss and serial versus uneven two-rank inference.
Raw local logs: `temp/th2_migration_20261026/diagnostic_tests.log`, `helper_tests.log`.
GPU gate passed: exact keep-all identity and causal prefixes; maximum FP32
cached/full logit difference 0.000212; all KV groups preserved. Initial PPL:
dense 17.7753; 6/8 static 26.7351, raw norm 21.7584, projected norm 21.2005,
joint 20.7038; 4/8 joint 55.8841. Full result/interval report is archived.
The method retains quality better than magnitude references on this probe, but
does not match dense quality and has not demonstrated sparse execution or a
learned router. Do not confuse the completed diagnostic with model training.
