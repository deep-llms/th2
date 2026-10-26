# Current task

Workspace `/disk/thuat/efficient_llm`; independent efficient-LLM research project.
Report times in Singapore time (SGT, UTC+8). Design documents:
[PROJECT_IDEA.md](PROJECT_IDEA.md), [FIRST_ARCHITECTURE_HYPOTHESIS.md](FIRST_ARCHITECTURE_HYPOTHESIS.md),
[CONTEXT_BUDGET_PLAN_20261026.md](CONTEXT_BUDGET_PLAN_20261026.md), and current
[SELECTIVE_DECODE_PLAN_20261026.md](SELECTIVE_DECODE_PLAN_20261026.md).

## Current result and scope

The user requested further research and B200 tests. The authorized context-budget
screen is complete: preserve every head's local attention and give 4/8 GQA groups
full causal history. Sink=4, recent window=128, original absolute positions.
Frozen original 28-layer Qwen3-0.6B-Base; no backbone training or environment change.
The previous 13 artifacts were downloaded and verified; its commands were restored to #0.
The user has now authorized the next selective-execution/retrieval screen (below). [Full results](CONTEXT_BUDGET_RESULTS_20261026.md).

- Previous head-removal diagnostic and matched-router results:
  [JOINT_GROUP_RESULTS_20261026.md](JOINT_GROUP_RESULTS_20261026.md),
  [ROUTER_PILOT_RESULTS_20261026.md](ROUTER_PILOT_RESULTS_20261026.md).
- Context study: 1,024 × 2,048 training prefixes, 131,072 teacher-feature examples,
  512 native HF Trainer updates/eight passes for matched joint/independent routers.
- 256 reused development documents × 2,048 targets: dense PPL 18.9674,
  learned joint 19.1899, independent 19.2066, fixed groups 19.3578.
  Learned joint is 1.17% above dense, versus 23.71% for previous head removal.
- 32 validation documents × 8,192 targets: dense 14.9860, joint 15.3083,
  independent 15.3270, fixed groups 15.5134. Joint is 2.15% above dense.
- Static execution, batch 8/context 8k: fixed groups 1.142× speedup with 3.52%
  higher PPL; fixed layers 1.560× with 11.45% higher PPL. Group splitting is slower
  at the other tested batch/context settings. No universal speedup.
- Learned/oracle quality references compute both branches: no selective-execution
  timing for learned routers yet. All KV remains stored. No KV-size-saving claim.
- Closely related prior work already exists. Joint-target gain is small; next
  decision should concern efficient execution/retrieval preservation, not a large
  adaptation sweep. Natural-text PPL is not downstream or retrieval accuracy.

## Current authorized next screen

Implement inference-only local/global selective decode, reuse the exact fitted
routers/model, and test correctness, whole-model speed and distant retrieval on
B200. No new model/router training, installation, downloads or old-output cleanup.
Plan: SELECTIVE_DECODE_PLAN_20261026.md. Compare native dense and dense using the
same kernel, so an implementation gain cannot masquerade as a routing gain.
CPU tests cover Triton interpreter math/cache behavior and seeded lookup construction.
Review also fixed a pre-existing cache-owner check: retain a unique object token,
not a recyclable Python object ID. Switching policy on a live cache must fail.

Planned run: 6y3o-efficient-selective-decode-20261026-a01, immutable inputs/source,
session efficient_selective_20261026_a01. Mandatory compiled FP32/BF16 gate before
eight-GPU profile and retrieval jobs. Reuse verified supervisor for Accelerate
copy/env, exact-worker burn reclaim, free-GPU checks and automatic burn restoration.
Launch/source 9481bee acknowledged at 23:26 SGT: pinned preparation passed for
128 lookup cases; Accelerate copy verified and environment confirmed eight B200s/BF16.
Initial compiled GPU gate passed (FP32 model max abs 3.82e-5).
The eight-GPU profile then stopped on rank 4: joint BF16 cached mean NLL difference
0.015731 exceeded the predeclared .01 limit. No retrieval ran; no profile accepted.
Burns restored 23:27:53 SGT and independently healthy 23:28:33; workers 599948–599955,
launcher 599879, source under selective-decode run; guard released.
Next: focused eight-document BF16 diagnostic with FP32-attention control, native
route replay, and FP32 computation at the same BF16-rounded weights. Original gate
is not relaxed. Run 6y3o-efficient-decode-numerics-20261026-a01, session
efficient_numerics_20261026_a01. Prior files preserved; no training.
78 local CPU tests passed. Read-only health export is the next action.

## Previous verified execution and GPU ownership

Execution remote `git@github-share:deep-llms/th2.git`, branch main; node
`thiennh-p6-6y3o-worker-0`, eight B200s. Dev repository also serves execution:
never repush unchanged executable commands.sh. Code `/mnt/local/deep-llms_th2`;
assets/outputs are separate. Earlier migration was verified; do not repeat cleanup.

Context run `/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-context-budget-20261026-a01`.
Launch/source `81ef234`; immutable source under run/source. Supervisor session
`efficient_context_20261026_a01`. Asset/probe hashes passed before reclamation;
Accelerate config copied and `accelerate env` confirmed eight GPUs/BF16.
Only freshly verified prior burn workers were stopped; all GPUs were verified free.

All three jobs exited zero by **19:03:33 SGT, 26 October 2026**. GPU numerical gate
passed; 512-step training/evaluation and decode timings passed. Native router
checkpoints 256/512 include weights, optimizer and scheduler. 74 local tests passed,
including actual two-rank old/new Trainer fits. FP32 teacher/static-reference
identity error zero; cache/causality maximum absolute error 0.0001593 on real weights.
No claim of bitwise BF16 full/cached equivalence or tested resume.

Automatic communicating burns restored/verified **19:04:39 SGT**; independently
healthy **19:09:24 SGT**, guard released. Workers **597076–597083**, launcher
**597007** at that observation. Eight ready ranks, collective probe sum 36,
advancing completed collective payload (through 377.52 GiB/rank in inspected tail).
Burn source `<context run>/source/resources/llm_pretrain_burn.py`, SHA256
`2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174`.
Reinspect PID/start identities before any future stop. Supersedes prior burn owners.

## Runtime, assets and evidence

Train Python `/mnt/local/conda-py311/envs/train/bin/python3.11`:
PyTorch 2.14.1+cu130, Transformers 5.9.0, datasets 4.8.5, Accelerate 1.13.0.
Model `/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6`.
Dataset `/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513`.
Trusted contract `resources/cx_english_20261020.json`. Environments unchanged.
No node outbound network: controller #d/#i for inputs, #2 exports.
Dropbox label 6Y3O, existing credentials/map in ignored DOT/temp. Never print.

Read-only verification `768740f`; export `4074c91` requests 13 artifacts including
compact routers. Raw receipts: temp/context_launch01.log, temp/context_health02.log.
Raw local archive temp/context_results_20261026 includes learned-router weights.
Durable reports: docs/results/context_budget_20261026/. All 13 remote SHA256s,
source snapshot, exact prior probe identity, sample counts, masks and metrics
validated. There are 23 shared 2k/8k validation documents; zero overlap with fit
documents. SHA256s pinned in the durable archive/artifact-hashes.json.
Prior raw results/weights remain in temp/router_results_20261026; prior durable
reports under docs/results/router_pilot_20261026. Full native checkpoints/features
remain on temporary B200 storage; do not claim they are locally backed up.
