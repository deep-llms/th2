# Current task

Workspace `/disk/thuat/efficient_llm`; independent efficient-LLM architecture project.
Report times in Singapore time (SGT, UTC+8). Design: [PROJECT_IDEA.md](PROJECT_IDEA.md),
[FIRST_ARCHITECTURE_HYPOTHESIS.md](FIRST_ARCHITECTURE_HYPOTHESIS.md),
[CONTEXT_BUDGET_PLAN_20261026.md](CONTEXT_BUDGET_PLAN_20261026.md), and current
[SELECTIVE_DECODE_PLAN_20261026.md](SELECTIVE_DECODE_PLAN_20261026.md).

## Active authorized work

User approved the sink/window and compact scheduling screen in
[WINDOW_COMPACT_PLAN_20261027.md](WINDOW_COMPACT_PLAN_20261027.md).
Implement/test an opt-in compact kernel and compare zero/four sinks with fixed
group assignments. Reuse frozen model and routers; no training. Preserve the
prior failed BF16 native acceptance gate. CPU tests and B200 preparation underway;
next launch must use the existing supervisor and verify burn handoff on any exit.

## Previously completed work

The user approved continuing execution/retrieval tests after the context-budget
screen. Implemented an inference-only split-K local/global decode kernel; reused
frozen original 28-layer Qwen3-0.6B-Base and exact fitted routers. No new training,
installation, download of external inputs, driver change or old-output cleanup.
All requested measurements are complete, artifacts verified, commands.sh #0.
No new job is queued. [Full results and caveats](SELECTIVE_DECODE_RESULTS_20261026.md).

- 80 local tests across maintained suites. Compiled GPU math/mask/poisoned-cache
  checks passed. Fixed a cache ownership bug using a retained unique object token.
- Initial BF16 cached native-comparison gate failed (.015731 nats on one 32-position
  document versus .01 limit). The first queue stopped correctly and restored burns.
- Follow-up: all eight FP32 trajectories at the same BF16-rounded weights agree
  within 2.87e-5. BF16 joint NLL mean delta +.000420, 129/7168 router choices differ,
  8/256 next-token choices differ. Native SDPA numerical sensitivity also appears
  in controls. Original gate remains failed; kernel remains **experimental**.
- Diagnostic-only whole-model timing at batch8/context8k: native dense 32.72 ms,
  joint kernel 24.26 ms, **dense same kernel 20.74 ms**. Learned routing is 16.9%
  slower than dense same-kernel. It is also slower than native at smaller cases.
  All kernel timings are conditional research measurements, not BF16 acceptance.
- Native SDPA retrieval, fixed synthetic balanced eight-choice lookup: dense,
  fixed groups, joint and independent routers each 128/128; local-only 45/128
  (32/32 near, 13/96 distant). Simple/correlated sanity check; saturated among
  global-capable methods, not broad downstream parity. No selective-kernel score.
- No KV-capacity savings: all groups remain cached. Prefill remains reference.
  Do not expand router training based on this screen. The next useful efficiency
  question is matched dense execution in an attention-bound serving workload.

## Earlier results

[JOINT_GROUP_RESULTS_20261026.md](JOINT_GROUP_RESULTS_20261026.md),
[ROUTER_PILOT_RESULTS_20261026.md](ROUTER_PILOT_RESULTS_20261026.md),
[CONTEXT_BUDGET_RESULTS_20261026.md](CONTEXT_BUDGET_RESULTS_20261026.md).
Head removal (6/8): joint PPL 23.4644 vs dense 18.9674. Local/global (4/8 global,
sinks=4/window=128): joint 19.1899 vs dense 18.9674 on 256×2k reused development docs;
joint 15.3083 vs dense 14.9860 on 32×8k docs. 23 shared short/long validation docs,
zero fit/validation overlap. Matched joint/independent routers used 512 native
Trainer updates/eight passes; backbone frozen. Small joint gain over independent
is not a novelty or practical efficiency claim.

## Verified GPU ownership and runtime

Execution/development remote `git@github-share:deep-llms/th2.git`, branch main.
Node `thiennh-p6-6y3o-worker-0`, eight B200s. Code `/mnt/local/deep-llms_th2`;
assets/outputs separate. Migration completed earlier: do not repeat cleanup.
Never repush unchanged executable commands.sh.

Latest run `/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-selective-research-20261026-a01`.
Launch/source `8adb821`, session `efficient_research_20261026_a01`.
Both measurement jobs exited zero at **23:52:48 SGT, 26 October**; 81.5s profile,
43.1s retrieval. Completion means measurement success, not passing numerical
acceptance (`diagnostic_only=true`, `native_nll_gate_passed=false`).

Accelerate copied/byte-verified and env confirmed eight-GPU BF16. Pinned inputs
verified before stopping exact known burn workers; all GPUs checked free.
Automatic burns restored/verified **23:53:59 SGT**, independently healthy
**23:54:51 SGT** in health commit `8d0efe0`; guard released. Eight ready ranks,
collective probe 36, advancing cycles/payload through 88.83 GiB/rank in inspected tail.
Workers **602209–602216**, launcher **602140** at observation. Burn source under
latest run's `source/resources/llm_pretrain_burn.py`, SHA256
`2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174`.
Reinspect PID/start identities before stopping. Supersedes older burn records.

Python `/mnt/local/conda-py311/envs/train/bin/python3.11`: PyTorch 2.14.1+cu130,
Transformers 5.9.0, datasets 4.8.5, Accelerate 1.13.0. Environments unchanged.
Model `/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6`;
data `/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513`.
Contracts `resources/cx_english_20261020.json`, `resources/context_fit_20261026.json`.
No node outbound network: controller #d/#i inputs, #2 exports.
Dropbox label 6Y3O, existing credentials/map in ignored DOT/temp; never print.

## Evidence

Initial gate/source `9481bee` under `6y3o-efficient-selective-decode-20261026-a01`;
numerical diagnosis/source `25dc043` under `6y3o-efficient-decode-numerics-20261026-a01`.
Full final export `b48013a`: all eight files independently SHA256 verified, exact
source/model/router/probe identity, 48 profile records, 640 scored lookups, 128
frozen lookup tensors, no missing/duplicate cases; labels balanced per cell.
Raw: temp/deploy_health01.log, temp/numerics_health01.log, temp/research_health01.log;
archives temp/deploy_failure_20261026, temp/numerics_results_20261026,
temp/research_results_20261026. Durable JSONs: docs/results/selective_decode_20261026/.
Earlier context weights/results: temp/context_results_20261026 (all 13 files verified)
and docs/results/context_budget_20261026/. Prior context source 81ef234, run
6y3o-efficient-context-budget-20261026-a01. Router weights reused byte-identically.
Full native training checkpoints/features remain only on temporary B200 storage;
do not claim they are locally backed up.
