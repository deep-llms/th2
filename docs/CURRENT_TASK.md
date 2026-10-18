# Current task

## Replacement node environment installation — 2026-10-18

The user confirms q359 is dead and requests an installation command pushed to
`deep-llms/th2` for the replacement node. **Do not monitor Dropbox until the user
asks.** The replacement hostname, GPU identity and Dropbox mapping are not yet
supplied; do not reuse q359-specific launch/preflight assumptions.

The execution checkout is `/disk/thuat/DOT-th2-inventory`, main. Its command is
`#i envs/train.txt envs/eval.txt +a`, job
`th2-dot-new-node-install-envs-20261018-a01`. This requests fresh Python 3.11
conda environments named `train` and `eval`. Both retain the existing model/data
library pins and add the old B200's validated FA4 4.0.0b33 (CUDA 13), CUTLASS DSL
4.8.0 and Quack 0.6.5. Eval uses the project's lm-eval 0.4.13 pin with its HF
extra and SentencePiece 0.2.2. The files are self-contained package lists.

Only installation is submitted. Installation success and CUDA functionality
remain unverified until the user requests a check. No training, download,
process stop, or automatic monitoring is queued. The old grouped verification
below has no retrieved final result; do not treat it as completed. DOT's source
commands.sh remains inactive; the execution checkout carries the install action.

## Grouped expert execution — implemented; remote verification incomplete, 2026-10-18

The user authorized implementing grouped GEMM and checking correctness and
complete-update throughput. `bdot/grouped.py` groups Q/O/MLP projections using
the pinned PyTorch CUDA BF16 primitive. `bdot.expert_backend` selects `reference`
(unchanged default) or `grouped`; original parameters/state names are preserved.
Only occupied experts enter autograd, preserving absent gradients for unused
parameters. CPU reference-kernel checks pass; real CUDA verification follows.

q359 read-only preflight `614c5d5` confirms the existing eight known communicating
burn workers and matching environment. Planned bounded verification reuses the
established supervisor, Accelerate installation/check, exact-PID reclamation,
free-GPU verification and automatic burn restoration. Output root:
`/mnt/local/_outputs/deep-llms_th2/q359-dot-grouped-20261018-a03/`.
Initial launch `ff5452b` passed projection/model gradient checks but stopped at
an overstrict fresh-AdamW coordinate tolerance (max 0.000600 at LR 0.000300).
The unchanged CPU BF16 reference independently reproduces the near-zero-gradient
first-step sensitivity. The revised probe records whole-update RMS drift and
checks identical existing moments separately. Initial supervisor restored and
verified all eight burns at 21:54:37 UTC. a02 passes real grouped projection and
all eight same-route SDPA/FA4 model cases (identical logits/losses; max relative
gradient difference 0.003245). A seeded fused-AdamW counter-device bug in the
DDP test fixture then stopped that suite; model code was unchanged. Launch
`ec19fe8` fixes the fixture and requires verified a02 burn restoration before a
fresh a03 preflight/launch. a03 passes CUDA numerics and all three checkpointed
two-rank BF16 probes (no-reuse, soft, home-only). Eight-rank A/B resume and
full-update benchmarks were submitted; their current status is unverified. Latest retrieved snapshot: 22:07 UTC;
local evidence `temp/grouped_a03_summary1.json`. Initial numerical evidence is
also in `temp/grouped_a02_summary1.json`.
At 22:20 UTC, Dropbox/controller status still ends at 22:07:25 UTC. Export
request `12881a3` and read-only live-status request `098061b` have no returned
receipt yet. No AWS/controller error has been returned, so the cause of the
missing updates is unknown. Do not infer completion, failure, or live burns.
The submitted supervisor restores/verifies burns on success or failure, but its
final handoff is not yet observed. Do not resubmit the GPU job. Retrieve the
pending receipt first; final source/results publication must use `#0`.
See [grouped verification notes](BDOT_GROUPED_GEMM_20261018.md).

User-authorized runner retry at 22:54 UTC: pushed `5c90b84` to th2 main,
verified the remote head, and submitted fresh read-only job
`th2-q359-dot-grouped-status-20261018-a02`. Its commands.sh includes an added
blank line, unique heartbeat/timestamp, existing result reads and bounded
nvidia-smi queries. It does not launch or stop GPU work. As of 22:58 UTC the
latest Dropbox listing and fresh `_RUN_STATUS_.log` download still end at
22:07:25 UTC; the retry commit has no returned acknowledgement. Local evidence:
`temp/grouped_retry_2257_list.jsonl` and `temp/grouped_retry_2257_status.log`.
Execution commands.sh remains this harmless read-only probe; DOT's source
commands.sh remains #0. Do not infer the current training/burn state from this
missing response, or resubmit the GPU verification job.

No pilot training is part of this optimization task. Existing scientific-review
fixes below are included in the version to be verified.

## Scientific-code review fixes — 2026-10-18

The user authorized fixing the verified findings in
[BDOT_PHASE_A_CODE_REVIEW_20261018.md](BDOT_PHASE_A_CODE_REVIEW_20261018.md).
All three main findings and four minor issues are addressed locally:

- Bypass forced-choice routers, leaving their gradients absent so AdamW cannot
  decay weights or apply momentum. Mask forced rows out of the z-loss numerator;
  retain the global all-token-step denominator.
- Periodic validation logs target and current-curriculum policies, without
  exploration. `eval_spike_loss` uses curriculum loss; final evaluation uses
  target-window hard routing.
- Separate optimization-objective labels from token-weighted LM aggregates,
  including checkpointed cumulative sums and logging cursors. Add interval
  throughput and peak GPU memory metrics.
- Align standalone/study boundary-batch routing and RNG; retain the GPU guard
  after an uncertain burn reclamation; specify matched D for the runtime gate.

The original 100-update CPU reproduction now stays at **0% off-default** with
z-loss enabled (previously **52.9419%**). Local regression and distributed test
evidence is recorded in [the fix receipt](results/bdot_phase_a_fixes_20261018.json).
The spec and implementation guide document these explicit policy amendments.
No remote job, process stop, deployment or research training was submitted.
DOT's commands.sh remains #0.

Before a pilot, revalidate the changed training paths on B200/FA4. The earlier
GPU evidence below predates these fixes. Strict resume remains enabled: earlier
code's checkpoints are not silently migrated to this changed objective/state
schema. Decode/prefill performance measurement remains separate unfinished work.

## Route-study implementation — historical, before the fixes above

The user requested the Phase-A route-study exporter, then asked to check
[BDOT_PHASE_A_CODE_REVIEW_20261018.md](BDOT_PHASE_A_CODE_REVIEW_20261018.md).
The exporter now extends the existing packed evaluator: one model/data load,
paired as-trained/fixed/random full-validation losses, 4,096-sequence route
statistics, a 256-sequence NPZ artifact and identity-checked sample comparisons.
It supports unpadded distributed evaluation. See
[usage](BDOT_IMPLEMENTATION.md#route-study). Local validation passed: **32 B-DOT
tests and 8 native evaluation tests**, including two-rank CLI coverage, an empty
rank and FA4 adapter wiring against a CPU reference kernel. See
[evidence](results/bdot_route_study_review_20261018.json). No B200 commands,
deployment or research training were submitted.

The review initially reproduced finding 1: home-only
curriculum A plus z-loss gives 46.6309% off-default evaluation routes at update
50 and 52.9419% at 100; the zero-z-loss control stays at 0%. The review follow-up
records the optimizer/normalization details needed for a fix. The evaluation
policy ambiguity and objective-vs-LM training-log labeling are also confirmed.
The exporter work did not change training; the later fixes above address these
findings.

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

- Revalidate pinned-runtime FA4/DDP/resume after the scientific-review fixes.
  Earlier GPU tests passed for the preceding code version.
- Measure decode/prefill speed separately; training throughput is not inference
  throughput.
- Run the implemented route exporter on actual pilot checkpoints for Gate 0a.
- Pin the actual dataset/tokenizer and record the validation split/manifest.
  Existing English cx_sampled_old on B200 can be reused; preserve its old
  training/validation split if selected. No new download is inherently needed.
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
