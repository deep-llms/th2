# Project notes

Status: context-compiled memory offline pilot; no scientific experiment results yet.

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

## Purpose and success criteria

### Overnight follow-on authorization — 2026-09-13

After the initial Step-1-only launch, the user authorized an unattended handoff:
verify preparation -> train the 4B-token common seed-17 model -> verify it ->
compile offline tables -> restore communicating burns. See
`OVERNIGHT_PILOT_20260913.md`. This does not launch reader adaptation/continuation
arms. Scientific source is preserved; new scripts add orchestration and artifact
gates only. There are still no scientific results. Ten new CPU/mocked handoff
tests passed, including exact stage order, failed-data blocking, PID reuse,
PID-1 refusal and live-child preservation.

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
