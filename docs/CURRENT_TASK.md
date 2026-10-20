# Current task

## Authorized overnight baseline workflow — 2026-10-20

The user authorized: install a separate older-PyTorch/FA2 environment; compare
it with production FA4 and the locally built FA2+C++20 environment; select a
correct, fast runtime; download only `nht10/cx_sampled`'s English
`qwen3_0.6b_base_en_30B` subset; run needed training/resume checks; launch the
original native Qwen3 architecture with 12 layers for 5,000 optimizer updates.
The design remains `docs/BDOT_v2_spec.md`. This baseline has no routers/FiLM and
is randomly initialized, as the spec requires. No further research arms are
authorized. All eight GPUs are used for complete-update timings and training.

Tie preference: FA4 approximately stock FA2+older PyTorch, both ahead of the
patched-source FA2 build. A speed win must exceed repeat variability and 5%
on native full updates without hurting the relevant routed-case throughput.
Every eligible choice must pass numerical checks; the selected choice must
also pass distributed training/checkpoint/resume verification before training.

Keep train/eval/train_fa2, system CUDA and the NVIDIA driver unchanged. New env:
`train_fa2_t28`, torch 2.8.0 / CUDA 12.8 and official FA2 2.8.3.post2 wheel.
Check compiled SM100 support and imports before GPU use. If incompatible, leave
it unused and compare FA4 with the fixed FA2 build, per the user's correction.
Use the established verified-burn handoff and restore burns on completion/error.

Read-only preflight `8b56289` verified 6Y3O at 01:03:55 UTC: protected imports
pass; known guard/burn hashes match; eight original burn workers remain active;
zero volatile uncorrectable ECC. The new environment is absent. This replacement
node has no project dataset/model, so tokenizer/config assets must also be
downloaded through the controller. Source commands.sh stays inactive.

Proposed recipe: spec sequence 1,024, 524,288 input tokens/update, seed 0,
EOS/isolation/reset positions, AdamW 3e-4 and 2% warmup. An optional question is
pending about full schedule vs cutoff. If unanswered, preserve continuation:
full 30B-token schedule, stop_after=5000. Record exact schedule, data revision,
preprocessing and validation split before launch; reuse the pool's existing
validation split rather than holding out training documents a second time.

Execution checkout: `/disk/thuat/DOT-th2-inventory`, th2 main. Current study root:
`/mnt/local/_outputs/deep-llms_th2/6y3o-attention-study-20261020-a01/`.
Local evidence: `temp/attention_study_20261020/`.

Runtime update, 01:35 UTC: stock FA2 2.8.3.post2 installed offline in
train_fa2_t28 after SHA256 verification. torch 2.8.0 / CUDA 12.8, C++11 ABI,
SM100 and compiled FA2 imports pass. Protected train/eval/train_fa2 snapshots
remain identical. The controller cache lacked the wheel; initial #datasets
request failed before creating a destination. Established #models single-file
request 16adf56 succeeded, then f8f9b75 performed the isolated offline install.

Comparison 4e1b1f6 completed successfully at 01:48:24 UTC. FA4 selected:
mean native update 0.615090 s, stock FA2/torch2.8 0.626876 s, patched FA2
0.610053 s. All nine numerical checks and all twelve timing jobs passed.
All eight enhanced communicating burn workers 12643–12650 were verified,
with matching live ownership and guard released, by 512d624 at 01:48:31 UTC.
See `docs/results/attention_comparison_20261020.json` for the compact receipt.
The selected-runtime distributed resume and real-data smoke gates remain.

Plan: `docs/ATTENTION_BASELINE_LAUNCH_20261020.md`. Implementation committed in
f8f9b75: explicit public FA2 backend sharing FA4's document-layout transport,
comparison/asset/queue helpers, local 44 ML regressions + seven old-torch CPU
adapter checks + four selection tests + queue preparation contract passing.
Committed `scripts/native_baseline_pipeline.py` prepares exact Trainer
caches while burns remain active, then gates the verification/baseline queue.
Its handoff test passes; committed in 7495e85.

Submitting the English-only download after selection/burn verification:
`temp/attention_study_20261020/download_commands.sh`, 3,016 exact single-file
#models URLs (3,008 English files, two pool manifests, six tokenizer/config
files), pinned revisions and unique destination paths. Model weights are not
needed for random initialization. Full schedule defaults to 57,220 steps,
cutoff 5,000, warmup 1,145; optional schedule question remains unanswered.

Download update (~02:14 UTC): the initial single-file list would take many
hours because of controller per-file overhead. Its exact task
`2026-10-20_01-51-37` was stopped (41647a8); existing files retained. Repartitioned
into eight disjoint shard groups, preserving all 3,016 requested inputs.
Batch commands: ignored `temp/attention_study_20261020/download_batch_0.sh`
through `_7.sh`. All eight batches acknowledged by 02:17 UTC.

| Index | Commit | Controller task ID |
|---|---|---|
| 0 | 398324c | 2026-10-20_01-58-08 |
| 1 | 7675f26 | 2026-10-20_01-59-36 |
| 2 | 25edec4 | 2026-10-20_02-01-18 |
| 3 | 8d623e9 | 2026-10-20_02-08-49 |
| 4 | 2276f60 | 2026-10-20_02-09-50 |
| 5 | 1ee6e34 | 2026-10-20_02-11-59 |
| 6 | 5c8756a | 2026-10-20_02-14-37 |
| 7 | 28ff772 | 2026-10-20_02-16-05 |

Controller per-task progress lines can be stale. Read-only node inspection
15322e4 at 02:06 UTC showed groups 0/1/2 all receiving files despite their stale
controller lines. All six tokenizer/config SHA256s passed. All eight verified
burns 12643–12650 remained at 100% utilization, guard released. No node/process
damage observed. Source: `temp/attention_study_20261020/corpus_arrival01.log`.
A local archive fallback was briefly prepared, then its exact owned CPU process
was stopped and its output removed when concurrency was confirmed. Nothing was
uploaded to HF; no dataset source was changed.

Next: monitor all eight and verify actual
file arrival/completion. Then use the reviewed
`temp/attention_study_20261020/launch_baseline_commands.sh` to run
`scripts/native_baseline_pipeline.py` under production train in a tmux session.
Fresh root: `/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a01`.
Pipeline: verify every dataset/model SHA256 and document count → prepare native
12-layer jobs → CPU-only exact Trainer cache fill while burns continue → require
predecessor verified burn ownership → fresh inventory/Accelerate copy+env check
→ supervised eight-rank resume test → real smoke25/resume50 → fresh baseline5000.
The GPU supervisor always restores and verifies communicating burns on exit.
Keep working until baseline training actually starts; do not end at download
or cache preparation. No other research arms are authorized.

At 02:21:54 UTC, node inspection f668079 observed 463/3,008 data files at
expected size (~21.36 GB), every group advancing. Node has 192 CPUs, unrestricted
CPU quota, ~1.96 TB MemAvailable. Baseline preprocessing is fixed to 128 workers
(before any real cache/training), leaving CPU capacity for the node; this value
is saved in the data recipe and must match future runs. Cache-helper CPU smoke
using SDPA metadata passed twice with identical train/validation recipes and
cache reuse (2,052 train sequences, 64 validation). FA4 numerical checks remain
the separate completed B200 evidence. Initial local FA4 cache-fixture attempt
failed only because local train_env has no FA4 package metadata; no production
code change was needed. Source log: `cache_helper_sdpa_test.log`.

## Isolated FA2 installation complete — 2026-10-20

The user authorized installation from official GitHub/PyPI inputs while
preserving B200 6Y3O stability. FA2 **2.8.3.post2** is now installed in
`/mnt/local/conda-py311/envs/train_fa2` against torch **2.14.1+cu130**.
The wheel was built offline from the SHA256-verified PyPI source with
CUDA 13.0.88, C++11 ABI enabled and SM100 only. The successful build took
1,625 seconds (~27 minutes), using four compiler jobs and one NVCC thread;
peak sampled build-process RSS was 16.55 GiB. No GPU workload was launched
or stopped, and no production backend or driver was changed.

The first source attempt (`6d551cb`) failed on C++17 versus the newer torch
headers. Retry `2f8b97e` changed exactly four setup.py compiler flags to C++20,
without kernel-source edits. `c11e34a` added the missing `libcudart.so` linker
symlink inside train_fa2 only. `scripts/install_fa2_offline.py` now includes
both build compatibility fixes for reproducibility. The env text installs
prerequisites; FA2 itself still requires this separate offline wheel build.

Read-only final verification `d09dc58` completed at **00:14:41 UTC**:
compiled FA2 fixed/variable-length forward/backward interfaces import in a
clean subprocess; train/eval and existing train_fa2 package metadata are
unchanged. Production imports including FA4 pass. All eight original burn
workers 500–507 under launcher 433 remain active at 98% GPU utilization;
volatile uncorrectable ECC and cgroup OOM/kill counts are zero.

Artifacts: `/mnt/local/_outputs/deep-llms_th2/6y3o-fa2-build-20261019-a02/`
contains `build.log`, `result.json`, `linker_name.json`, and the built wheel.
Wheel SHA256: `a34f9f06f824a60e233df06dc60668526b97c3cdf12cf00bbc39478008c9aadd`.
See [installation receipt](results/fa2_install_success_20261020.json) and
[environment/build notes](FA2_B200_ENVIRONMENT_20261019.md) for local log paths,
checksums, exact wheel tag, and upstream support evidence.

**Remaining:** GPU numerical, throughput and resume tests are not performed
by installation/import verification. Do not adopt FA2 for production based
only on these receipts. Source and execution commands.sh return to #0;
no build, test or new workload remains queued by this task.

## Node health after the failed FA2 request — 2026-10-19

The user emphasized preserving B200 stability. Read-only inspection `161d804`
completed at **23:27:14 UTC** on 6Y3O. All eight GPUs respond with unchanged
UUIDs and driver 580.167.08, zero volatile uncorrectable ECC counts, and the
same GPU PIDs 500–507. Their parent 433 runs `/tmp/llm_pretrain_burn.py` through
system Python 3.10; all eight GPUs remain 98% busy. No processes were signaled.
`train` and `eval` match the earlier checked versions and again pass core/FA4
imports. Host available RAM is about 1.83 TiB and local free disk about 24.1 TiB;
cgroup OOM/kill counters are zero. No sign of damage was found in these checks.
This is not a full hardware stress test or a pre-install file comparison.
See [health receipt](results/b200_6y3o_health_20261019.json).

No further install, build or GPU test was launched. commands.sh returns to #0.
Future FA2 work must stay in `train_fa2` and a dedicated build directory,
preserve production environments/system CUDA/drivers, and bound compiler
parallelism after checking memory/disk availability.

The user also identified the official PyPI FA2 source distribution. Dev-only
inspection verified its published SHA256 and bundled CUTLASS header; the
GitHub tag source archive omits that header. Both setup scripts include SM100
build flags. An offline source build is another possible route, subject to
toolchain/dependency verification; an operator-supplied wheel is not the only
option. This corrects the overly restrictive earlier wording below. No source
archive has been downloaded to B200 and no source build has been submitted.

## FA2 installation failure diagnosed — 2026-10-19

The user requested the reason for the failed FA2 install. Read-only discovery
`6bb2024` located `/mnt/local/_env_status/train_fa2.status.json`; inspection
`d3434c5` at **23:18:04 UTC** returned its exact failure:
`result: missing`, `source: gap`, `overall: incomplete`. Its sole gap is
`flash_attn-2.8.3.post2+cu13torch2.14cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`.
Other requested packages report success. The node cache contains no matching
FA2 wheel. A fresh official release asset check also finds no matching wheel.
This confirms an unavailable requested prebuilt dependency, before FA2 kernel
execution; the controller's upstream wheel-store contents were not audited.

The install specification named the runtime-compatible artifact without first
establishing its availability. Reusing that prebuilt request requires a matching
torch 2.14.1 / CUDA 13 / Python 3.11 / C++11-ABI FA2 build made available through
the operator's wheel store; see the source-build alternative above. Do not
repeat the same failed request, substitute
an incompatible wheel, or change production torch/FA4 environments.

See [failure receipt](results/fa2_install_failure_20261019.json) and
[environment notes](FA2_B200_ENVIRONMENT_20261019.md). Local sanitized diagnostic:
`temp/dot_th2_6y3o/fa2_failure_detail_20261019_a01.log`. No reinstall, source
build, training or process stop was performed. Source and execution commands.sh
return to #0 with this diagnosis publication.

## Replacement-node Dropbox checks authorized — 2026-10-19

The user clarified that Dropbox checks may resume freely now that the new
6Y3O folder is available. This supersedes the earlier monitoring deferral.
The controller reports installation success for `train` and `eval` at
`e2dabb0`; the separate `train_fa2` request at `960bbc9` failed with exit 3.
Ordinary log export `2efa752` could not retrieve installer details because
there were no normal job logs yet. No installation was repeated.

Read-only inspection `5f06832`, job
`th2-6y3o-attention-env-inspect-20261019-a01`, completed on
`thiennh-p6-6y3o-worker-0` (started **2026-10-19 23:05:47 UTC**). All three
environments have the expected torch 2.14.1, CUDA 13.0, Python 3.11.15 and HF
pins; core imports pass. `train` and `eval` also import FA4 4.0.0b33 successfully.
Eval has lm-eval 0.4.13 installed; its task execution was not tested.
`train_fa2` has **no flash-attn package**, so installation is incomplete and
FA2 testing remains blocked. At this stage the installer error was unknown;
the subsequent diagnostic above establishes the missing-wheel result.

The node exposes eight B200s with driver 580.167.08 and 183359 MiB each. All
eight had active GPU processes; their identities were not checked or changed.
This inspection did not allocate tensors, test kernels or launch training.
Verified package/import data and raw-log hashes:
[receipt](results/b200_6y3o_environment_20261019.json). Raw files:
`temp/dot_th2_6y3o/runtime_20261019_a01.log` and `status_20261019_a03.log`.
Source and execution commands.sh return to **#0** with this receipt publication.

The subsequent diagnostic above resolves the installer error. Preserve
production environments and driver. CUDA numerical, throughput and resume
verification on this node remains separate future work.

## Isolated B200 FA2 environment — installation request, 2026-10-19

The user authorized creating `train_fa2` for a later FA2 comparison without
modifying the current training environment. Submit only
`#i envs/train_fa2.txt +a`, job `th2-fa2-isolated-env-20261019-a01`, to th2.
The Python 3.11 spec uses `fresh: false`, retains the existing torch/HF pins,
and requests the exact FA2 2.8.3.post2 CUDA 13 / torch 2.14 / C++11 ABI wheel.
The upstream release lacks this wheel; internal runner availability is unknown.
If missing, the operator must supply a matching build. Do not change torch or
reuse the local torch 2.8 wheel merely to complete installation.

See [installation and verification notes](FA2_B200_ENVIRONMENT_20261019.md).
Production environment specs, model code and drivers are unchanged. The request
failed and its partial runtime was inspected as recorded above; no GPU tests,
training or process stops are queued. The earlier Dropbox deferral is superseded
by the authorization above. Both commands.sh copies return to #0 after inspection.

## Local FA2 comparison — completed, 2026-10-18

The user asked whether stable FA2 should replace FA4, then requested testing on
the development machine only. FA2 2.8.3.post2 passes the local A100 kernel,
native-Qwen and routed B-DOT numerical, causality/isolation and activation
checkpointing checks. Two full-update timing repeats per model/backend give
FA2 throughput gains versus dense isolated SDPA of 11.22% for native Qwen,
9.76% for matched D, and 1.01% for hard mixed routing. The routed difference
is small relative to variation. All twelve runs completed; used GPUs are free.

This uses an isolated package overlay and the existing local torch 2.8.0+cu126
test runtime. Production model/backend code, train_env and drivers are unchanged.
The test reuses the FA4 packed-layout transport but executes the real FA2 kernel;
receipts label that substitution explicitly. There is no public FA2 training
option yet. This is not B200 performance, distributed/resume or long-run evidence.
See [report](FA2_LOCAL_VERIFICATION_20261018.md) and
[receipt](results/fa2_local_20261018.json). No new push or remote action occurred;
th2 still has the installation commit `e2dabb0`. Dropbox remains unchecked.

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
