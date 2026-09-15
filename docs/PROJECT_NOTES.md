# Project notes

## 28L/12B Phase-A launch — 2026-09-15

Phase A of `ccm_28layer_scaleup_plan_v3_12B.md` submitted as one queue:
extension data preparation, both post-hoc 12L Shallow follow-ups, and the 28L
seed-17 10B common training with the decided smoke policy. The 28L panel,
12L locked validation and replication seeds require separate authorization.
Pre-launch decisions are recorded in `SCALEUP28_PRELAUNCH_DECISIONS_20260915.md`;
the implementation contract in `SCALEUP28_IMPLEMENTATION.md`; the queue
protocol in `SCALEUP28_PHASE_A_20260915.md`.

## Seed29 automatic replication / explicit Delta policy — 2026-09-14

User authorized all remaining seed29 offline phases, reusing its completed 4B
common checkpoint. They explicitly chose the same Delta thresholds applied
independently to seed29, rather than seed17's decision controlling every seed.
This inclusion-policy amendment is recorded before seed29 memory outcomes;
primary hypotheses, thresholds, training, data and model remain unchanged.
See [the queue protocol](SEED29_REPLICATION_20260914.md). Stage2 always uses
fresh modules/optimizer and the common checkpoint, not Stage1 learned state.
No seed43 or locked validation is automatically scheduled.

## A3 joint frequency × variance — 2026-09-14

Completed on all five seed17 Stage2 step3815 checkpoints, full 20M-input-token
D_dev each, unchanged normal memory behavior. Shared compile-defined 5×5 bins
cover 12,942,338 identical hit targets. Contextual has lower NLL than Base,
Isolated and Shuffled in all 25 cells, but higher NLL than Grad in all 25.
V5 has larger Contextual gains than V1 against all three fixed-table controls
within each coarse frequency band; this is not a causal variance result.
All original segment/population losses replay exactly, both original marginal
bin summaries match, and all 13 result-file hashes pass local verification.
No per-cell CIs or regression fitted, no new training/validation/seed29 arms.

See [A1–A3 results](MEMORY_DIAGNOSTICS_RESULTS_20260914.md) for all cell counts,
four contrasts, limitations and execution evidence. Local A3 artifacts:
`outputs/a3_results_20260914_a01/results/`. Launch `0ae7e33`; exact final
archive-part export `842181c`. Completion 19:53:03 UTC; original all-eight burns
verified at 20:04:12 UTC with eight ranks in one NCCL communicator. Current
burn owner/handoff flag is recorded at the top of CURRENT_TASK.md.

## Memory dependency diagnostics — 2026-09-14

See [the complete A1/A2 report](MEMORY_DIAGNOSTICS_RESULTS_20260914.md).
Completed full seed-17 Stage-2 D_dev evaluations for Contextual, Isolated, Grad,
normal and memory-off. Normal replay matched original losses exactly on every
segment/population. Disabling memory worsens all three models and makes each
worse than Base-Continue (all overall paired 95% CIs positive).
Off-minus-normal NLL: Contextual +0.003841, Isolated +0.000888, Grad +0.009034.
Mean per-token contribution/hidden norm ratios: 5.82%, 2.01%, 8.68%.
The memory branch remains useful at inference; this does not establish a
causal decomposition, specifically deep advantage, replication, or system
superiority. Grad remains best in normal NLL. No new training or D_val access.
Full results/paired records are local and hash-verified. The workflow finished
at 19:07:25 UTC and restored original all-eight NCCL burns; its later heartbeat
confirmed one active worker per GPU at 98% utilization.

Status: seed-17 Stage 1, Stage 2, and A1/A2 diagnostics completed; seed-29
common pretraining completed. No seed-29 memory comparisons or seed-43 run yet.

## Stage-1 dev result and next phase — 2026-09-13

All six 977-update training runs, seven dev evaluations and artifact gates
completed successfully at 17:16:25 UTC. All-eight-GPU original burns were
verified again at 17:47:51. Evidence: local
`temp/stage1_handoff_child_recheck_20260913_a01.log` and
`temp/stage1_status_child_recheck_20260913_a01.json` (fresh flattened Dropbox
child-folder exports). Earlier stale-status observations below are historical.

| Stage-1 arm | Overall dev NLL | PPL |
|---|---:|---:|
| Base (unadapted) | 3.242478541 | 25.597087 |
| Contextual | 3.240683974 | 25.551192 |
| Isolated | 3.241132248 | 25.562649 |
| Shuffled | 3.241994293 | 25.584694 |
| Shallow | 3.240763330 | 25.553220 |
| Delta | 3.240717930 | 25.552060 |
| Grad | 3.239475761 | 25.520339 |

Contextual minus Isolated: -0.000448274 nats/token, document-bootstrap 95% CI
[-0.000468182, -0.000428767]. Contextual minus Shuffled: -0.001310319,
CI [-0.001342067, -0.001279312]. Both primary directions are favorable but
effects are small; Grad is better. These are single-backbone development
diagnostics, not multi-seed evidence or a training-efficiency claim.

User authorized the planned Stage-2 seed-17 continuation: Base, Contextual,
Isolated, Shuffled, Grad, 3815 updates each from the same 4B common checkpoint
with fresh optimizers/modules, followed by dev evaluations and burns.
Delta fails the preregistered overall-NLL safeguard and is not included.
No locked validation or replication launch. See `STAGE2_RUN_20260913.md`.

## Stage-1 sequential launch — 2026-09-13

User authorized six seed-17 reader-adaptation arms. Launch `18c89f6`:
Contextual → Isolated → Shuffled → Shallow → Delta → Grad, each on all eight
GPUs for 977 steps / 256114688 tokens, with the backbone frozen. The sequence
then evaluates Base and all six arms on dev, validates outputs, calculates the
two primary contextual contrasts, and restores verified original burns.
See `STAGE1_RUN_20260913.md`. No Stage-2/val launch is authorized by this job.

Startup verified: input artifacts passed; old idle observer exited; only its
verified burn workers were stopped; two 30-second free checks passed. First
Contextual launch at 14:49:30 UTC. Export `dfe0c85` shows 131 finite contiguous
updates / 34340864 tokens, the correct 8-rank bf16 seed-17 contract, and the
same common checkpoint hash. These are startup checks, not a quality result.
Outputs: `/mnt/local/_outputs/deep-llms_th2/ccm_stage1_seed17_20260913_a01`.
79 local tests passed; no research-core changes or old-output cleanup.

## Original B200 compilation completed — 2026-09-13

Fresh #2 export `0bac963` confirms compilation exited 0 at 14:10:26 UTC:
1,000,000,000 tokens, 17849.15 seconds reported wall time (4h57m29s),
5926582414 output bytes. All five table constructors passed the artifact gate
at 14:11:48, including corpus/vocabulary/checkpoint identity checks. The
handoff checked all eight GPUs free and verified burn startup before recording
all Steps 1–3 complete at 14:13:01. No reader-quality results yet. Evidence:
`temp/b200_handoff_fresh_20260913_1416.log` and
`temp/b200_workflow_fresh_20260913_1416.json`.

## Compiler performance benchmark — 2026-09-13

Common seed-17 training completed and passed verification (4B tokens, 15259
updates; approximately 3h25 wall time). Offline compilation is the next active
stage. A user-approved dev-only benchmark found substantial avoidable CPU
accumulation and padding overhead. Full details and caveats are in
`COMPILER_BENCHMARK_20260913.md`.

On a fixed 398818-token CulturaX sample with a random full-size writer, median
pass times were 19.534s for current CPU accumulation, 6.334s for GPU accumulation
at the same batch size, 3.255s with length grouping, and 2.036s on four A100s
including NCCL merging. Counts matched exactly. Same-batch GPU accumulation
produced identical bf16 lookups on this sample; changed batch layouts produced
small but nonzero bf16 forward differences. These are implementation benchmark
results, not scientific quality results or B200 throughput predictions.
No production job was stopped or changed and nothing was pushed for the test.

Follow-up four-A100 equivalence audit: replaying identical captured states with
NCCL merging, or distributing unchanged original batches, reproduced all three
bf16 lookup tables exactly on this sample. Master statistics still differ by
floating-point rounding. Length grouping did **not** reproduce the tables:
up to 0.7852% relative L2 mean difference, with differences already present in
the forward residuals. Do not treat the benchmark's 1% guard as a scientific
equivalence threshold. Prefer preserved physical batches when reproduction
is the priority; their four-GPU speed was not measured by the 9.60x result.
See the follow-up section of `COMPILER_BENCHMARK_20260913.md` and
`temp/compiler_equivalence_audit_20260913/complete.json`.

## Local implementation — 2026-09-12

The binding sources are Section 13 of the frozen design and the v2 clarification
answers. The implementation is in `ccm/`; see `PILOT_IMPLEMENTATION.md`.
No previous sparse-embedding or Stagewise model/data split is inherited.
The approved corpus is verified CulturaX English raw data; new document-aware
role/segment manifests and the pinned Qwen3-0.6B-Base tokenizer are required.

Initial local testing covers a tiny CPU end-to-end offline pipeline, bf16
equivalence and optimizer-master checks, and two-process CPU/Gloo gradient
equivalence. These are correctness tests, not production performance evidence.
The real 12-layer shape has 344,354,816 parameters and tied input/output weights
(verified on the meta device without allocating a real model).

Review fixes include stronger checkpoint/table contracts, exact record checksums,
paired comparison step/writer/seed checks, an explicit locked-held-out reporting
path, and final-evaluation identity based on the checkpoint metadata contract
rather than model-weight hash alone. No B200 workload was changed.

Statistical-policy choices are explicit CLI arguments. Do not silently substitute
the suggested shared/content-hash bootstrap for the v2 source-document/independent
policy, or apply a Delta replication policy without recording the selected flags.

## Additional review — 2026-09-13

59 CPU tests and two-process CPU/Gloo reference checks passed. Local four-A100
CUDA/NCCL tests also passed at full pilot dimensions (12 layers, length 2048,
262144 slots), including frozen/trainable table semantics, bf16 activation
checkpointing, replica equality and save/reload. The tiny GPU pipeline covers
all offline constructors and required arms. No B200 processes or jobs changed.

Reporting now enforces the final lock's bootstrap policy and the scientific
10000-replicate requirement; invalid compiler batch settings fail before work.
See `IMPLEMENTATION_REVIEW_20260913.md` for exact evidence and remaining gates.
These are engineering checks, not scientific results or throughput claims.

## B200 real-data smoke — 2026-09-13

See `B200_SMOKE_20260913.md`. Verified all 50 English raw shards (122167447875
bytes) against pinned official CulturaX metadata and their actual B200 hashes.
Verified the pinned official Base config/tokenizer; created and validated a
fresh 2752512-token engineering corpus with no reuse of the old sampled split.

The actual trainer passed eight updates / 2097152 real input tokens on eight
B200s. All eight tiny GPU pipelines and five full-capacity systems checks passed,
including NCCL replica equality and frozen/Grad semantics. Local suite: 62 tests.
No scientific model-quality result is claimed. Raw short-run timing was order-
dependent enough that it must not be used for comparative speed claims.

Original communicating burns were restored and separately verified after the
handoff exited: all eight at 98% utilization, 2510 MiB/device at 01:00 UTC.
Small results were pulled through `#2` and all archive/member hashes verified.
Full scientific data preparation, compilation-cost measurement and recovery
planning remain; no full-budget pilot was launched.

## Full scientific data preparation — 2026-09-13

User authorized step 1 only. th2 commit `a4facbd` submits
`scripts/prepare_pilot_data.sh`: offline CPU preparation using the default locked
PILOT quotas, followed by data validation, top-262144 bigram vocabulary, dev-only
coverage and final artifact/batch validation. No training, GPU termination,
cleanup or automatic table-capacity change is included. Remote startup at
01:36:51 UTC passed node/environment/assets/full-budget/disk checks. Preparation
is in progress, not yet completed; see `CURRENT_TASK.md` for current observations.
At the 01:40 UTC snapshot, 49000 documents had been scanned and 38839996 tokens
written. The raw-source checksum stage had passed before tokenization began.

Fresh data: `/mnt/local/_data/deep-llms_th2/ccm/pilot_v1_20260913_a01`.
Reports: `/mnt/local/_outputs/deep-llms_th2/ccm_prepare_pilot_v1_20260913_a01`.
Compile/adapt are part of the common-training corpus, not extra optimization
data. Existing raw files and older sampled/smoke artifacts are preserved.

## Overnight follow-on authorization — 2026-09-13

After the initial Step-1-only launch, the user authorized an unattended handoff:
verify preparation -> train the 4B-token common seed-17 model -> verify it ->
compile offline tables -> restore communicating burns. See
`OVERNIGHT_PILOT_20260913.md`. This does not launch reader adaptation/continuation
arms. Scientific source is preserved; new scripts add orchestration and artifact
gates only. There are still no scientific results. Ten new CPU/mocked handoff
tests passed, including exact stage order, failed-data blocking, PID reuse,
PID-1 refusal and live-child preservation. The full 72-test local suite also
passed. Commit `0750f56` was verified ARMED on B200 at 02:32:50 UTC, with
preparation progressing and all original burns active; no training yet.
The final local suite passed 75 tests. The independent dev monitor's complete
first retrieval was verified (four files and local SHA256), with a fresh remote
02:42:52 UTC heartbeat: 1896188056 prepared tokens, all eight original burns
still 98% / 2510 MiB. The B200 supervisor and dev hourly monitor are both active;
training/compilation are gated future stages, not completed results.

## Purpose and success criteria

Describe the actual problem, working hypothesis, baseline and decision criteria.

## Infrastructure

Record non-sensitive repository/branch roles, active machines, GPU allocation,
runtime paths, environment versions, and Dropbox labels. Keep credentials,
private folder URLs and operator instructions in ignored local files.

## Reproducibility

Record input revisions/manifests, preprocessing, seeds, configuration, resource
budget, metrics and validation criteria. For ML projects, distinguish training
from held-out results, document batch/precision/scheduler settings, and retain
checkpoint/evaluation provenance. Do not claim speed from tiny smoke tests.

## Results and decisions

| Date (UTC) | Run / commit | Evidence location | Result | Decision |
|---|---|---|---|---|

## Operational lessons

Record confirmed failure causes and validated recovery procedures; distinguish
project exceptions from runner/infrastructure failures. Avoid live-status claims
without a timestamp. Do not treat a failed run as successful because a PID ended.

## Backup / portability

List which small result/config artifacts are retained locally, which large
artifacts remain on temporary remote storage, and the authorized backup method.
Keep shared code and docs on the development repository rather than only in a
temporary execution worktree. Track project guides except the local-only
`AGENT_GUIDE.md` and `DROPBOX_ACCESS.md`; never track populated
credentials or `temp/INSTRUCTION.md`.
