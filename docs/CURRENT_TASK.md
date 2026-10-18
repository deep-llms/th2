# Current task

## B200 verification in progress — 2026-10-18

The user authorized publishing DOT to th2 and running B200 verification.
Source was published at `5d1f037`; the controller blocked its preflight before
execution on a redundant test assertion mentioning a Hub publication option.
Removed only that assertion; the actual offline network-denial test remains.
Fresh read-only preflight `9b52375`, job `th2-q359-dot-preflight-20261018-a02`,
uses `/mnt/local/_outputs/deep-llms_th2/q359-dot-verification-20261018-a02/`.
Do not infer environment validity or GPU verification from submission alone.

The next authorized stage is synthetic numerical/FA4/backward/NCCL and complete
Trainer-update verification, followed by automatic verified burn restoration.
The old q359 supervisor is reused with its PID-handle helpers and seven passing
CPU ownership/handoff tests; its obsolete standalone reclamation CLI was not
copied. `resources/accelerate_b200.yaml` is the previously validated eight-GPU
configuration, distinct from the generic single-process example. Copy to the
selected interpreter's actual Accelerate cache and verify before reclaiming
known burns. Research pilot training and corpus downloads are not part of this
verification. Local receipts: `temp/bdot_b200_verification_20261018/`.

The prepared suite retains failed independent checks and exits unsuccessfully
if any required check fails. It reports natural near-tie route changes separately
from same-route kernel checks using explicit separated diagnostic router biases.
Neither diagnostic changes production model initialization. Throughput uses the
full 12-layer models, eight ranks, microbatch16, accumulation4, length1024,
524,288 input tokens/update, BF16, checkpointing off, 3 warmup +10 measured
updates per native/D/mixed-hard/mixed-soft case, separately for SDPA and FA4.

## Previous implementation and setup evidence

Updated 2026-10-18. The Phase A foundation is implemented and CPU-validated;
local A100 numerical, NCCL and SDPA throughput checks have also been run.
No research training or dataset download has been launched. A read-only remote
project inventory has now completed as described below.

Design: [BDOT_v2_spec.md](BDOT_v2_spec.md), **v1.3**, and
[review response](BDOT_v2_review_response.md). The current architecture is a
12-layer Qwen3 model trained from scratch; older architecture descriptions are
superseded. Later phases remain conditional on Gate 0a.

## Implemented

- Native Qwen3 dense reference and routed middle with step-owned K/V, FiLM,
  fixed/learned/random choices, reuse on/off, exact feasibility tables,
  curricula A/B, balancing and correctly normalized router loss.
- Reuses train.py, HF Trainer/Accelerate and strict checkpoint/resume. Adds
  only the model-specific Trainer behavior; dynamic unused experts use DDP.
- Frozen validation document IDs and deterministic document stream, EOS,
  isolated attention and reset positions. Continuous packing drops only the
  stream's final remainder; the original baseline packing remains available.
- Configurations D, P1-A/B, R, N-A/B; sequential queue preparation, packed PPL,
  route interventions, native lm-eval wrapper and a full-update benchmark tool.
- Recreated the outdated local train_env with the user's authorization.

See [implementation and commands](BDOT_IMPLEMENTATION.md) for scope and tests.
Initial local results are in docs/results/bdot_phase_a_local_20261018.json.
The follow-up review fixed nonpersistent buffer restoration during checkpoint
loading. Expanded checks (25 tests plus checkpointed hard/soft CPU DDP probes)
are in docs/results/bdot_phase_a_review_20261018.json. See
[GPU verification](BDOT_GPU_VERIFICATION_20261018.md) for subsequent CUDA evidence,
runtime limitations and a borderline default-SDPA AdamW check. Pinned FA4 backward
explicitly rejects this dev machine's A100 (SM80); forward-only results are separate.

## Remaining before a pilot

- Verify the assigned execution machine's environment/assets. Local CUDA
  tests used compatible older PyTorch builds, not train_env's CUDA 13 build.
- Validate the pinned runtime and FA4 backward on a confirmed compatible GPU;
  measure full-update throughput at the actual distributed pilot batch. Local
  mixed hard routing was 76.5% of current D / 61.7% of native dense. This is
  preliminary, not eight-GPU T8 clearance.
- Finish full route-study export/reporting before Gate 0a: 4,096-sequence path
  statistics, 256-sequence route artifact and cross-seed comparisons. Current
  code reports off-default share/reuse and supports fixed/random interventions.
- Pin/download the actual dataset/tokenizer and freeze its validation manifest.
  No scientific performance claim follows from the tiny local test fixtures.

## Assigned execution target and cleanup preparation — 2026-10-18

The user explicitly assigned the previous `deep-llms/th2` GitHub repository,
`main` branch, q359 B200 and its existing Dropbox folder to DOT. The user asked
for a separate minimal checkout containing only commands.sh and a first
read-only inspection of the B200 project directory.

Execution checkout: `/disk/thuat/DOT-th2-inventory`. Commit `4831f9a` descends
normally from previous head `624d63a`; its tree contains only commands.sh.
No force push or change to the local DOT/old-project source trees was used.
Job: `th2-q359-dot-project-inventory-20261018-a01`, `#1 +30+a`.
It inventories `/mnt/local/deep-llms_th2` without deletion, file-content dumps,
process stops, GPU work or environment changes. Inventory completed successfully
at 18:32:47 UTC on `thiennh-p6-q359-worker-0`; the returned script SHA256 matches
the submitted commit. There are 28 top-level entries: old source directories
and files remain, alongside controller logs/status files. Synchronization did
not remove old files deleted from GitHub. No B200 files have been deleted.
Commit `5e395c7` restores commands.sh to #0; GitHub still contains only that file.
Local evidence: `temp/dot_th2_inventory_20261018/` and
[verified receipt](results/dot_th2_inventory_20261018.json).
The user subsequently authorized cleaning this remote project directory.
Cleanup job `th2-q359-dot-project-cleanup-20261018-a01`, commit `1676c12`,
removes the inventoried old source entries and all resources except the known
live burn script. It preserves commands.sh, runner logs/status files and
resources/llm_pretrain_burn.py. It refuses unexpected paths, mounts, non-burn
GPU work or live project scripts and verifies GPU identities before/after.
Local fixture checks passed for scoped deletion, external symlink preservation,
runtime preservation, refusal paths and idempotence. Cleanup completed at
18:53:02 UTC: all 41 listed entries removed; all eight GPU burn workers and their
launcher retain the same PID/start-time identities. The only remaining entries
are commands.sh, _run_log_, _dl_active.txt, _install_active.txt,
_previous_run_status.log and resources/llm_pretrain_burn.py. Separate data,
model, output and conda directories were untouched. Execution commit `162455c`
replaces commands.sh with #0 and a comment, removing the destructive body.
Evidence: [cleanup receipt](results/dot_th2_cleanup_20261018.json) and
`temp/dot_th2_inventory_20261018/cleanup_a01.log`.

DOT's own commands.sh remains #0. Research training has not been authorized
or launched by this cleanup/inventory request.
