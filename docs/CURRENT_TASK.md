# Current task

Workspace `/disk/thuat/efficient_llm`; independent efficient-LLM architecture project.
Report times in Singapore time (SGT, UTC+8). Design: [PROJECT_IDEA.md](PROJECT_IDEA.md),
[FIRST_ARCHITECTURE_HYPOTHESIS.md](FIRST_ARCHITECTURE_HYPOTHESIS.md),
[CONTEXT_BUDGET_PLAN_20261026.md](CONTEXT_BUDGET_PLAN_20261026.md), and latest
[WINDOW_COMPACT_PLAN_20261027.md](WINDOW_COMPACT_PLAN_20261027.md).

## Latest authorized work completed

The user approved testing pure sliding-window attention and compact selective
scheduling. Frozen original 28-layer Qwen3-0.6B-Base and identical fitted routers
and fixed allocations; no training. [Results](WINDOW_COMPACT_RESULTS_20261027.md).

- All 82 local tests and the compiled GPU math/FP32 gate passed. Opt-in
  `triton_compact` launches no empty local tiles; local heads write final outputs
  directly. Original kernel remains a control; native SDPA remains default.
- Native SDPA PPL on the same 256 × 2k documents: dense **18.9668**, fixed
  first-four + recent-128 **19.3570**, same fixed groups with recent-128 only
  **75.5163**. All 256 documents worsen without sinks. Do not generalize this to
  trained sliding-window architectures or claim four sinks is optimal.
- B8/8k attention microbenchmark: dense original 86.12 us, fixed original 51.50 us,
  fixed compact 48.57 us. Dropping four sinks has negligible custom-kernel cost.
- Whole-model eager B8/8k: dense original kernel **22.346 ms**, fixed original
  **22.489**, fixed compact **22.656**, joint original **26.179**, joint compact
  **27.160**. Compact scheduling does not improve whole-model speed. Native SDPA
  dense is 32.728 ms; compare against the fastest dense control instead.
- FP32 original-weight cached checks passed. BF16 original/compact trajectories
  still differ on some individual documents by >.01 nats. Prior native acceptance
  gate remains failed. Profile is diagnostic-only, not backend acceptance.
- Keep sinks for this frozen policy; no adoption of compact as default. No cache
  capacity saving: all KV retained. No new training or follow-up queue authorized.

## Earlier evidence

[JOINT_GROUP_RESULTS_20261026.md](JOINT_GROUP_RESULTS_20261026.md),
[ROUTER_PILOT_RESULTS_20261026.md](ROUTER_PILOT_RESULTS_20261026.md),
[CONTEXT_BUDGET_RESULTS_20261026.md](CONTEXT_BUDGET_RESULTS_20261026.md),
[SELECTIVE_DECODE_RESULTS_20261026.md](SELECTIVE_DECODE_RESULTS_20261026.md).

Whole-group removal loses substantial quality: learned joint 23.4644 versus dense
18.9674 PPL on 256 documents. Preserving local attention reduces the joint router's
PPL gap to 1.17% at 2k / 2.15% at 8k. Joint versus independent supervision gives
only a small gain. Simple native-SDPA retrieval saturated: dense/fixed/joint/
independent 128/128, local-only 45/128. No practical learned-routing speed gain,
broad downstream parity, cache-capacity reduction, or novelty established.
Do not scale up router training based on these results.

## Verified GPU ownership and runtime

Execution/development remote `git@github-share:deep-llms/th2.git`, branch main.
Node `thiennh-p6-6y3o-worker-0`, eight B200s. Code `/mnt/local/deep-llms_th2`;
assets/outputs separate. Migration completed earlier; do not repeat cleanup.
Never repush an unchanged executable commands.sh.

Latest root `/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-window-compact-20261027-a01`.
Launch/source `36e30fa`, session `efficient_window_20261027_a01`; independent health
`b10b860`, export `4cf8964`. Gate 54.5s, profile 129.6s, quality 33.5s, all zero
exit by **01:19:46 SGT, 27 October**. Source/model/probe/router identity pinned.
Accelerate copied/byte-verified, eight-GPU BF16 env confirmed. Previous exact burn
workers 602209–602216 reclaimed, all GPUs checked free before GPU jobs.

Burns restored/verified **01:20:56 SGT**, independently healthy **01:21:17 SGT**:
eight ready ranks, collective sum 36, advancing 40 cycles / 44.41 GiB per rank,
guard released. Workers **604193–604200**, launcher **604124** at observation.
Burn path under latest root `source/resources/llm_pretrain_burn.py`, SHA256
`2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174`.
Reinspect PID/start identities before stopping. Supersedes all earlier ownership.

Python `/mnt/local/conda-py311/envs/train/bin/python3.11`: PyTorch 2.14.1+cu130,
Transformers 5.9.0, datasets 4.8.5, Accelerate 1.13.0. Environments unchanged.
Model `/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6`;
data `/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513`.
Contracts `resources/cx_english_20261020.json`, `resources/context_fit_20261026.json`.
No node outbound network: controller #d/#i inputs, #2 exports.
Dropbox label 6Y3O, existing credentials/map in ignored DOT/temp; never print.

## Evidence

Raw logs `temp/window_launch01.log`, `temp/window_health01.log`; downloaded files
`temp/window_results_20261027/`. Durable `docs/results/window_compact_20261027/`.
All 11 exports matched independent node SHA256s; source/input identity, 60 kernel
cases, 88 whole-model timing rows, 48 microbenchmark rows, 256 document pairs and
burn/completion receipts verified locally. commands.sh #0; no further queue.
Earlier verified artifacts remain under `docs/results/` and ignored temp archives.
Fitted router weights backed up in `temp/context_results_20261026/`; full native
router training checkpoints/features remain only on temporary B200 storage.
