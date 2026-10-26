# Current task

Workspace `/disk/thuat/efficient_llm`; independent efficient-LLM research project.
Report times in Singapore time (SGT, UTC+8). Design documents:
[PROJECT_IDEA.md](PROJECT_IDEA.md),
[FIRST_ARCHITECTURE_HYPOTHESIS.md](FIRST_ARCHITECTURE_HYPOTHESIS.md).

## Completed work and current scope

The user authorized the first diagnostic, then approved the matched 6/8-router
pilot and cached-decode profile. Both are complete. No adaptation, sparse-kernel
job, additional method or training continuation is queued. commands.sh is #0.

- First diagnostic: [protocol](JOINT_GROUP_DIAGNOSTIC.md),
  [results](JOINT_GROUP_RESULTS_20261026.md). Original frozen 28-layer pretrained
  Qwen3-0.6B-Base. Joint output reconstruction beats norm oracles at six groups
  but still loses to dense; four groups loses too much quality.
- Current completed pilot: [protocol](ROUTER_PILOT.md),
  [results](ROUTER_PILOT_RESULTS_20261026.md). Matched linear joint/independent
  routers, identical initialization/data/Trainer recipe, frozen backbone. 1,024
  train prefixes, 256 new validation prefixes, excluding the first diagnostic's
  128 evaluation documents. All use 2,048 next-token targets per document.
- 512 router-only HF Trainer updates, 131,072 cached feature examples, eight
  passes. Dense masked execution; no measured sparse speed or KV-size saving.
- New-probe PPL: dense 18.9674; joint oracle 22.0110; learned joint 23.4644;
  learned independent 24.1605. Joint wins on 250/256 documents, delta −0.029232
  nats/token, paired 95% CI [−0.032145, −0.026263]. Still 23.71% higher PPL than
  dense. Different held-out documents from the first diagnostic.
- Native SDPA cached-decode profile, batch 1/8 × context 2k/8k: idealized 4–15%
  speed headroom before router/dispatch cost. This is an eager HF estimate,
  not achieved sparse speed or an optimized-serving result. Review tradeoffs
  before expanding training or implementing a kernel.

## Verified execution and current GPU ownership

Execution remote `git@github-share:deep-llms/th2.git`, branch main; node
`thiennh-p6-6y3o-worker-0`, eight B200s. Development repository also serves as
execution repository: never repush unchanged executable commands.sh.
Code `/mnt/local/deep-llms_th2`; data/models/outputs are separate external roots.
Migration was verified at 09:05:45 SGT on 26 October; do not repeat cleanup.
See `docs/results/efficient_llm_migration_20261026.json` for archival evidence.

Pilot launch/source `5554c61`; read-only verification `d8e96c1`; export `fd97c19`.
Run root `/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-router-pilot-20261026-a01`.
Supervisor session `efficient_router_20261026_a01`; immutable source under root.
All three jobs exited zero by **15:56:08 SGT, 26 October 2026**. Accelerate config
was copied to its actual cache and `accelerate env` confirmed eight GPUs/BF16.
Asset hashes passed before burn reclamation; GPUs were verified free before jobs.

Automatic communicating burn restored and verified at **15:57:19 SGT**; live
ownership and advancing collective payload rechecked at **16:00:18 SGT**. Guard
released. Workers 594491–594498 under launcher 594422 at that observation.
Burn source: `<run root>/source/resources/llm_pretrain_burn.py`, SHA256
`2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174`.
Reinspect live PID/start identities before any future stop; never use name-based
kills. This supersedes the earlier joint-diagnostic burn ownership record.

## Runtime, assets and evidence

Train Python `/mnt/local/conda-py311/envs/train/bin/python3.11`:
PyTorch 2.14.1+cu130, Transformers 5.9.0, datasets 4.8.5, Accelerate 1.13.0.
Model `/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6`.
Dataset `/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513`.
Pinned hashes in `resources/cx_english_20261020.json`. Environments unchanged.
No node outbound networking: controller #d/#i for inputs, #2 for exports.
Dropbox label 6Y3O; existing credentials/map in ignored DOT/temp. Do not print.

67 local tests passed, including actual two-rank native Trainer fitting, bank
independence, data exclusions, cache/causality, profiling identity and ownership.
B200 correctness gate and trained-router reload/cache checks passed. Native
checkpoints 256/512 retain weights/optimizer/scheduler; both exist on B200.
No claim of tested resume or bitwise BF16 full/cached equivalence.

All ten downloaded artifacts matched remote SHA256 checksums. Probe hashes/counts,
exclusions, evaluation coverage, fixed recipe, source snapshot and burn handoff
were independently validated locally. Archive: `docs/results/router_pilot_20261026/`.
Full raw downloads and learned router weights: `temp/router_results_20261026/`.
Raw receipts: `temp/router_launch01.log`, `temp/router_health02.log`.
CPU logs: `temp/router_tests_final_20261026.log`, `temp/router_helper_tests_20261026.log`.
Full native checkpoints and cached teacher features remain on temporary B200
storage. Compact learned-router weights are backed up locally, outside Git.
