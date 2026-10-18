# Current task

## GPU FA4 resume verification — passed

The user explicitly requested the remaining B200 resume check. Launch `82f6428`
on q359 completed the real train.py/Trainer FA4 test on all eight GPUs. The
uninterrupted 200-step run and fresh-process restarts (100→200 and 20→40) match
exactly on all ranks: data, RNG, both routing branches, loss, model, AdamW,
scheduler and routing state. Maximum loss difference is 0.0. This used the
existing FA4 defaults and a 12-layer reduced-width synthetic fixture; no model
or attention-kernel change was needed. See [resume report](BDOT_FA4_RESUME_20261018.md).

Accelerate was copied and verified. All eight GPUs were free at 19:48:20 UTC
before the test. Supervisor and automatic burn handoff finished successfully at
19:53:15 UTC. Final read-only inspection `7c49881` at 19:56:35 UTC verified
eight active burns, advancing collectives and released ownership. The final
source/results publication uses #0. No research training is queued.

Remote root: `/mnt/local/_outputs/deep-llms_th2/q359-dot-resume-20261018-a01/`.
Local evidence: `temp/bdot_b200_resume_20261018/`.

## B200 verification complete — 2026-10-18

The user authorized publishing DOT to `deep-llms/th2` main and running GPU
verification on q359. DOT is published from `/disk/thuat/DOT-th2-inventory`;
local DOT remains the source directory and its commands.sh is inactive.
Launch commit `c3ce8d1` completed the synthetic suite: **16 required checks pass**.
This includes full-size native equivalence, real FA4 forward/backward,
checkpoint/isolation checks, two-rank NCCL accumulation and eight-GPU Trainer
benchmarks. One optional natural near-tie route comparison changes discrete
paths across backends; separate same-route checks pass. Keep the backend fixed.

Hard-routing throughput is 89.2% of matched dense D with SDPA and 85.9% with
FA4, exceeding the 60% runtime gate in this setup. Relative to native Qwen these
are 78.1% and 75.4%. These are short synthetic measurements, not pilot results.
See [B200 report](BDOT_B200_VERIFICATION_20261018.md) and
[machine-readable summary](results/bdot_b200_20261018.json).

The selected existing eval_fa4 environment matches the training pins. Accelerate's
resource config was copied bytewise and verified as eight-process BF16. Exact
known burn workers were reclaimed and all eight GPUs verified free before tests.
Supervisor restoration finished successfully at 19:27:32 UTC. Final read-only
inspection `d0a24cc` at 19:30:53 UTC confirms all eight communicating burns remain
live, collectives advance and ownership is released. Training losses are finite
and benchmark source hashes match. A parser-only issue in the preceding check
was fixed; no GPU tests were restarted. The final code/results push uses #0.

Remote artifacts:
`/mnt/local/_outputs/deep-llms_th2/q359-dot-verification-20261018-a03/`.
Local receipts: `temp/bdot_b200_verification_20261018/`.
No research pilot, corpus download, package reinstall or driver change occurred.
The existing environment's older lm-eval version was not used or validated.

## Previous implementation and setup evidence

Updated 2026-10-18. The Phase A foundation is implemented and CPU-validated;
local A100 numerical, NCCL and SDPA throughput checks have also been run.
No research training or dataset download has been launched. The inventory and cleanup history is recorded below; subsequent B200
verification is summarized above.

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

- Pinned-runtime backward, synthetic eight-GPU throughput and GPU FA4 resume
  checks passed. Revalidate if the pilot recipe changes materially.
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
