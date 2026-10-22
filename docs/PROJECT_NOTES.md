# Project notes

User preference (21 October 2026): report times and ETAs in **Singapore time
(SGT, UTC+8)**. Machine-readable receipts may retain their original UTC timestamps.


## Fixed-attention partial-MLP study — 22 October 2026

User approved six new 4,768-step seed-0 fits, reusing completed D0: learned and
random routing of 25%, 50%, or 100% of each middle layer's SwiGLU channels.
All attention and norms stay at their original depth; the remaining MLP
channels form a fixed core. Split original weights without extra backbone
parameters or different initialization RNG. Route only the selected slice,
with the unit-forward gate on that slice alone. Both learned/random controls
share curriculum A (10% home-only, then radius 1 with reuse allowed).
This is an exploratory follow-up authorized after the failed v3 pilot.

Reuse HF Trainer/Accelerate, packed English data, seed-0 order, FA4 isolation,
reset positions, EOS boundaries, batch 524,288, length 1,024, microbatch 16,
accumulation 4, warmup 96 and the full existing pilot LR schedule. No extra
seed, longer fit or dataset change. Preserve checkpoints 1,000/2,500/4,768
and latest, as in the preceding v3 queue.

Implementation **0bcadad**, launch **13f3940**, root
`6y3o-dot-mlp-20261022-a01`. All acceptance gates passed before the first arm
started **11:04:05 SGT**. Old CUDA behavior and D checkpoint loss are unchanged;
both learned-partial and random-full eight-rank resume tests are exact.
Full-size learned throughput is 94.5–98.3% of D; random controls 99.6–102.5%.
Benchmarks remove the learned router prior to exercise multiple experts;
research configs keep the original prior. This timing is not final wall time.
Numerical backend/split comparisons use BF16 tolerances, not bitwise claims.

Copied/verified Accelerate, verified burn-worker reclamation, eight free GPUs
and automatic success/failure burn restoration use the existing supervisor.
Verified local launch evidence: `temp/partial_mlp_20261022/evidence/` (37 files).
Health **6cb07e3** at 11:12:42 SGT confirms Partial-25 step 700, active routing,
finite gradients, declining validation loss and a complete step-500 checkpoint.
Source/data match. [Launch receipt](results/bdot_mlp_launch_20261022.json).
See CURRENT_TASK for progress. Preserve source during the running queue.

## C1 conditional v3 control — 22 October 2026

The user approved C1 seed 0, the v3 section 6.1 control triggered by V1 beating V2.
C1 copies C2 with only placement FiLM disabled: random/no-reuse/radius1,
curriculum none; unchanged 4,768-step FA4/grouped recipe, seeds and data order.
This comparison retains the plan's learned-versus-random curriculum difference;
it is not an isolated intervention on learned router weights.
Launch **98c430e**, fresh root `6y3o-dot-C1-20261022-a01`; first health **9bf815d**
confirms actual training and correct free-GPU/Accelerate/data handoffs.
No MLP-only variant, second seed or longer run is authorized by this follow-up.
C1 subsequently finished at **09:04:48 SGT**, loss 3.55846145 / PPL 35.10914.
V1 beats it by 0.014539 nats in seed 0 but remains 0.049459 above D and fails
another gate criterion (default-path gain). No second-seed confirmation is
triggered under the plan. All retained checkpoints and final optimizer state
are verified. Burns restored 09:05:54, verified live at 09:51:00 SGT.
See [completion evidence](results/bdot_C1_final_20261022.json) and
[updated v3 report](BDOT_V3_RESULTS_20261022.md).

## Research scope

[BDOT_v2_spec.md](BDOT_v2_spec.md) v1.4 defines the research. Implement Phase A
first: compute-matched bounded routing, absolute windows and step-owned K/V.
Do not build capacity/depth modes, other KV designs, STOP or re-entry until the
specified pilot gate passes. Configurations reject unsupported modes explicitly.

The plain dense reference is native Qwen3ForCausalLM with 12 layers and random
initialization. B-DOT reuses those Qwen3 components. The matched dense D has
FiLM; native-equivalence tests turn FiLM off. They are different references.

## Random-routing control R complete — 21 October 2026

User-authorized R trained from scratch with routing=random, curriculum=none,
reuse=true; otherwise identical P1-A source/data/validation and full 4,768-step
recipe. Launch 16dd3d5 verified Accelerate, exact burn reclaim and free GPUs.
Final inspection 0b891c1: finished **9:50:49 a.m. SGT**, loss **3.61016131**,
PPL **36.9720**, 9,704 sequences / 9,915,698 scored tokens; Trainer time 55m58.7s.
All 20 checkpoints and final optimizer/scheduler/eight-rank RNG state retained.
Burns restored **9:52 a.m. SGT**, still verified live at **11:54 a.m. SGT**
with collective progress and guard released. No further experiments queued.

R beats P1-A by 0.00578 nats but trails D by 0.11570. Both learned routing
curricula fail to beat the random-training control on this single-seed pilot;
this comparison also changes the routing curriculum. N-A remains closest to D
but fails its D + 0.01 quality threshold. Phase B has not been authorized.
Evidence: [R final receipt](results/bdot_R_final_20261021.json),
[combined results](BDOT_PILOT_RESULTS_20261020.md).

## Training contracts

- Keep HF Trainer/Accelerate, AdamW/scheduler and checkpoint/resume behavior.
  BDOTTrainer adds separate global router-loss normalization, one balance update
  per optimizer step and ordinary DDP unused-parameter handling. It rejects
  unvalidated FSDP/DeepSpeed/compile/reentrant-checkpoint paths.
- Model initialization follows the model seed. Routing draws use a separate
  checkpointed stream/counter and rank; evaluation never advances that stream.
- The v1.4 pilot uses the existing CulturaX English train/validation splits,
  ordinary HF map packing and one seed-0 packed shuffle with a sequential sampler.
  Keep preprocessing at 128 workers/map batch 1,000 to match the native baseline.
  A shared manifest checks exact text overlap once, drops contaminated validation
  sequences and fixes periodic/final selections. Legacy document_stream remains
  supported for old tests/checkpoints; it is no longer the pilot default.
- Append document EOS, isolate documents and reset positions per fragment.
  LM target tokens and router decisions have separate denominators. Validation
  loss excludes router terms. The inherited packed_shuffle mode is unchanged.
- max_steps defines the full schedule; stop_after is an absolute cutoff. Keep
  all checkpoints by leaving save_total_limit unset. Strict resume compares
  data, source, runtime, model and training receipts before loading state.
- Gate pilots complete 2.5B tokens rounded to 4,768 updates, warmup 96, cosine
  floor 10%, with no stop_after. LR and routing share the same horizon. The
  completed native 30B-schedule/cutoff-5k run remains a separate reference.
- Pilot queue default: D, P1-A, P1-B. R and the matching N follow explicit
  curriculum selection. The queue generator does not launch or select winners.

## Stage 0a launch — 2026-10-20

User authorized the minimal research queue after dispatch optimization.
Launch d24bf55 on 6Y3O prepares and verifies the pinned English pool, then
runs T12 → T14 → D → P1-A → P1-B under the existing GPU supervisor.
All three pilots explicitly use FA4/grouped experts, seed 0 and the complete
4,768-step / 96-warmup schedule; ordinary HF cache/order is shared. R/N await
curriculum selection. No Phase B or extra seeds are queued.

All 3,008 input hashes pass; exact raw-text overlap scan finds zero validation
documents in training. Frozen validation contains 1,024 periodic / 9,704 final
sequences. Accelerate config was copied and checked with accelerate env;
known burns were reclaimed by identity, all eight GPUs were empty before tests.
T12 has exact state and zero loss differences on every rank. Full-width T14
passes: training gap 0.003070 and validation gap 0.004109 nats (<0.01).
D began at 19:51:44 UTC; automatic burn restoration covers completion/failure.
See [launch evidence](results/bdot_pilot_launch_20261020.json) and CURRENT_TASK
for live-state timestamp and output paths. The old native 5k run remains a
separate long-schedule reference, not this pilot's D.

The queue completed successfully at 22:46:31 UTC. All D/P1-A/P1-B runs reached
4,768 steps and retained 20 checkpoints each, including final weights,
optimizer, scheduler and eight rank RNG states. Full-validation loss/PPL:
D **3.49446352 / 32.9326**, P1-A **3.61594091 / 37.1863**, P1-B
**3.63270401 / 37.8149**. Both routed arms miss Gate 0a's D + 0.01-nat quality
threshold. R/N and route interventions remain unrun; no additional experiments
were launched. Burns restored automatically at 22:47:42 UTC and verified live
at 23:05:08 UTC on all eight GPUs, with supervisor guard released.
Final exports are hash-verified locally in `temp/pilot_final_20261020/`;
see [completion evidence](results/bdot_pilot_final_20261020.json).

After reviewing these results, the user authorized one matched N-A run with
reuse forbidden, keeping P1-A's curriculum and full recipe. This asks whether
repetition/omission contributes to the regression through training; it is not
an inference-time route intervention. This authorization does not expand to
R, Phase B or additional seeds. See CURRENT_TASK for launch state.
Launched as **0bde7f9**, fresh root `6y3o-dot-N-20261020-a01`, with source and
data/validation recipe equality verified against P1-A before GPU reclamation.
The established supervisor copied/verified Accelerate, reclaimed identified
burns, confirmed all eight GPUs free and started N-A. Initial optimizer progress
is verified; completion and final burn restoration remain pending. Evidence:
[N launch receipt](results/bdot_N_launch_20261020.json).

N-A completed 4,768 steps at 2026-10-21 00:40:24 UTC: full-validation loss
3.52599809, PPL 33.9877, Trainer time 59m24s, 20 checkpoints retained. Reuse is
zero and off-default routing is 24.93%. Its loss gap to D is +0.03153 nats,
smaller than P1-A's +0.12148, but it still misses the D + 0.01 quality bar.
Burn restoration and current identities were verified before further work.
The user now explicitly authorized R after N completion: random routing,
reuse on, curriculum none, otherwise the same seed-0 4,768-step recipe.
See [N final evidence](results/bdot_N_final_20261021.json) and CURRENT_TASK.

## Route study and scientific-review decisions

`eval/packed_ppl.py --route_study true` reuses packed evaluation for full selected
validation losses under as-trained/fixed/random routes. It collects a fixed
4,096-sequence diagnostic prefix and exports 256 sequences by default. Cross-seed
agreement requires matching exported token identities and evaluation contracts;
it is explicitly sample-scoped. The comparison can use all 4,096 by requesting
that artifact size for both runs. Fixed/random interventions are not replacements
for separately trained D/R controls. See the implementation guide for usage.

The user subsequently authorized the 2026-10-18 review fixes. Whole forced-choice
routers are bypassed (`grad=None`, including at the optimizer boundary during
the home-only phase); forced rows in mixed batches contribute zero z-loss.
Normalization deliberately remains **all input tokens × routing steps**, summed
over accumulation and ranks. AdamW momentum/decay must not update an inactive
router. The original reproduction changes from 52.9419% off-default at update
100 to 0%, with the default z-loss coefficient unchanged.

Periodic eval_loss is now the curriculum loss at completed updates / full
schedule, with target hard-routing loss logged separately; both omit exploration.
Transitions evaluate both phases on identical weights and reset the spike minimum.
Gate 0a uses curriculum loss; final studies use the target policy.
Native HF objective aggregates are explicitly labeled separately from
token-weighted LM loss. Cumulative LM sums and logging cursors are checkpointed;
interval wall throughput includes validation/checkpoint overhead. Peak memory
is the maximum across ranks. The v1.4 training runtime gate compares against native
Qwen, with matched D timing also reported.

Standalone B-DOT evaluation and route studies visit all packed batches, including
all-boundary batches, so route counts and random-route streams agree for the
same layout/seed. No-reuse and curriculum CPU DDP/accumulation tests cover the
new loss semantics. See [resolution](BDOT_PHASE_A_CODE_REVIEW_20261018.md#resolution-after-authorized-fixes)
and [fix receipt](results/bdot_phase_a_fixes_20261018.json). GPU/FA4 validation of
these changes is still needed before a pilot. Strict source/state resume checks
remain in force; do not bypass them to resume checkpoints made before this fix.

## Runtime and evidence

The local train_env was recreated on 2026-10-18: Python 3.11.15, PyTorch 2.14.1,
Transformers 5.9.0, Accelerate 1.13.0, Datasets 4.8.5, Tokenizers 0.22.2,
Safetensors 0.8.0 and lm-eval 0.4.13. Direct requirements are in envs/train.txt
and envs/eval.txt. No other environment or NVIDIA driver was changed.

The installed CUDA 13 wheel cannot use this dev machine's older driver. Initial
tests used CPU. Subsequent local GPU tests use separate compatible runtimes;
the pinned production-runtime/backend checks remain separate launch gates.
See [BDOT_IMPLEMENTATION.md](BDOT_IMPLEMENTATION.md) and
[local validation receipt](results/bdot_phase_a_local_20261018.json).
Historical template GPU evidence does not validate this new model.

The 2026-10-18 follow-up review fixed HF checkpoint loading of derived routing
buffers: regenerate no-reuse quota tables and zero update accumulators after
meta-device materialization, without changing persistent state. Tests now cover
all pilot checkpoint types, CPU BF16, independent dispatch math and checkpointed
hard/soft DDP. See [review receipt](results/bdot_phase_a_review_20261018.json).

Local GPU verification on 2026-10-18 used free A100 GPUs, without changing the
driver or train_env. SDPA model/gradient/checkpoint tests and two-rank NCCL pass;
see [GPU report](BDOT_GPU_VERIFICATION_20261018.md) for the default-SDPA AdamW
roundoff caveat and measured throughput. Native dense is also reported because
the current fixed D shares generic dispatch overhead. Pinned FA4 4.0.0b33 has
an SM80 forward path but rejects SM80 backward. Do not bypass that kernel guard
or claim FA4 training validation from forward-only evidence.

## Infrastructure

On 2026-10-18 the user explicitly assigned the previous `deep-llms/th2` GitHub
repository (`main`), q359 B200 and its existing Dropbox folder to DOT. The
separate execution checkout is `/disk/thuat/DOT-th2-inventory`. Its initial
normal descendant commit `4831f9a` replaces the tracked old-project tree with
only commands.sh and submits a read-only directory inventory. Previous history
and the local old-project checkouts are preserved. This is not authorization to
delete datasets/checkpoints/environments or stop processes on the node.

The first inventory succeeded on `thiennh-p6-q359-worker-0` at 18:32:47 UTC.
The source tree retains 28 top-level entries, including old project code and
controller-managed logs/status files: this runner does not delete old source
files just because they were removed from GitHub. No remote deletion was
performed. Commit `5e395c7` returns the minimal execution tree's commands.sh to
#0. Evidence: [inventory receipt](results/dot_th2_inventory_20261018.json).

The user then explicitly authorized source-directory cleanup. Job `1676c12`
completed at 18:53:02 UTC, removing 41 old source/resource entries. Preserved
commands.sh, controller logs/status files and resources/llm_pretrain_burn.py,
which is used by the existing eight-GPU burn. All eight worker identities and
the launcher were unchanged; no signals, new GPU work or environment changes
occurred. Data/model/output/conda roots were not modified. Commit `162455c`
deactivates the command and removes its cleanup body. GitHub contained only commands.sh at that point. See [cleanup receipt](results/dot_th2_cleanup_20261018.json).

DOT itself currently has no initialized Git repository; its commands.sh is #0.
Actual research assets still require selection and verification. Assets and library
execution are offline; no weights/data have been downloaded for the research
run. Credentials and shared URLs remain outside Git. Existing approved Dropbox
credentials are read from the old development project's ignored temp directory;
the q359 folder mapping and downloaded receipts are in DOT's ignored
`temp/dot_th2_inventory_20261018/`.


## B200 verification — 2026-10-18

Published DOT to the authorized th2 main branch from DOT-th2-inventory. Preflight
`0f37b2c` verified q359 and the existing eval_fa4 interpreter: all training-library
pins match, with FA4 4.0.0b33 and CUDA 13. No reinstall/driver changes. The
lm-eval version is older than our eval pin and was not used for these checks.
Launch `c3ce8d1` copied/verified Accelerate's eight-GPU configuration and reclaimed
only verified burn workers. The numerical suite and eight-GPU benchmarks passed
all 16 required cases; one optional natural near-tie route comparison differed.
Keep the attention backend fixed for an experiment. Same-route comparisons pass;
this does not promise identical routing trajectories across kernels.

Hard mixed routing is 89.2% of matched D throughput on SDPA and 85.9% on FA4;
relative to native Qwen these are 78.1% and 75.4%. Both pass the 60%-of-D runtime
gate in this synthetic setup. Full report, limitations and receipts are in
[BDOT_B200_VERIFICATION_20261018.md](BDOT_B200_VERIFICATION_20261018.md).
No research pilot or corpus download was launched. The source code remains in
DOT; the separate Git checkout publishes it without tracking temp credentials.

The supervisor finished at 19:27:32 UTC and a separate live check at 19:30:53 UTC
verified eight communicating burns, advancing collectives and released guard.
See [handoff receipt](results/bdot_b200_handoff_20261018.json). commands.sh returns
to #0 with the final code/results publication; no unattended pilot is queued.


## GPU FA4 resume evidence — 2026-10-18

Launch `82f6428` ran a 200-step uninterrupted reference, then fresh-process
checkpoint continuations 100→200 and 20→40 through real train.py on eight B200s.
All ranks match exactly for data/RNG, both routing branches, loss, model, AdamW,
scheduler and balancing state. Maximum loss difference is zero. The fixture
uses 12 layers with reduced width/vocabulary, BF16 and the existing FA4 defaults.
No deterministic override or production model change was necessary. See
[resume report](BDOT_FA4_RESUME_20261018.md) and its all-rank JSON receipt.
The supervisor restored communicating burns on success. A final live check at
19:56:35 UTC verified all eight workers, advancing collectives and released guard.
commands.sh returns to #0. No pilot was launched.


## Grouped expert execution, 2026-10-18

Completed follow-up on 6Y3O, 2026-10-20: all CUDA/two-rank checks and both
eight-rank A/B resume suites passed; resume loss difference is zero and state
is exact. Two 50-update FA4 repetitions at seq1024/global524288 show mean
reference → grouped times of 0.807290 → 0.768397 s for hard routing (+5.1%
throughput), 1.026440 → 1.005195 s for soft (+2.1%). Native: 0.615911 s.
Grouped adds about 0.3 GiB/GPU. It is a modest opt-in optimization; default
remains reference, and backend changes across resume remain prohibited.
Tests finished at 14:52:44 UTC; burns restored/verified at 14:53:50 and independently
checked live at 14:55:56. See [completed comparison](BDOT_GROUPED_GEMM_20261020.md)
and `results/bdot_grouped_comparison_20261020.json`. Environment, driver, corpus
and existing baseline artifacts were unchanged. Historical initial attempt:

`bdot.expert_backend` now supports optional CUDA BF16 `grouped` projections;
`reference` remains the default. Original expert parameter names and unused
parameter/AdamW semantics are preserved. CPU tests (40 B-DOT, 49 utilities),
real B200 SDPA/FA4 numerical cases and three checkpointed two-rank BF16 tests
pass. Same-route fixture logits/losses match exactly; gradient relative error
is at most 0.325%, with explicit first-AdamW roundoff sensitivity recorded.

Eight-rank A/B resume and 50-update full-size timings were submitted under the
normal guarded burn handoff, but remote telemetry has not advanced past
22:07 UTC as of 22:20 UTC. Their outcome and final burn status are unverified;
no speed claim or default switch is justified yet. See
[verification notes](BDOT_GROUPED_GEMM_20261018.md). No pilot was launched.

## Replacement node environments — 2026-10-18

The user confirms q359 is dead. Submit only the controller installation directive
to th2: `#i envs/train.txt envs/eval.txt +a`. The two self-contained specifications
create fresh Python 3.11 `train`/`eval` environments with matching training pins
and the previously verified CUDA 13 FA4 4.0.0b33, CUTLASS DSL 4.8.0 and Quack 0.6.5.
Eval retains the project's lm-eval 0.4.13 pin, with HF support and SentencePiece.
Installation and GPU imports must be verified later; do not monitor Dropbox
until the user asks. Replacement node identity/mapping is pending, so old
q359 launchers and environment paths must not be reused without inspection.

On 2026-10-19 the user supplied the replacement node's Dropbox folder,
`deep-llms_th2_thiennh-p6-6y3o` (label `6Y3O`). Its shared URL is stored only in
ignored `temp/dot_th2_6y3o/dropbox_folders.txt`. Dropbox credentials are read
from `/disk/thuat/deep2shallow/temp/dropbox_credentials.txt`. Machine identity
and environments were verified later; see the 6Y3O entries below.

## FA2 local comparison — 2026-10-18

Stable FA2 2.8.3.post2 passes local A100 numerical/model/packing checks via a
test-only kernel substitution. In two repeats of full HF Trainer updates,
throughput versus dense isolated SDPA improves 11.22% for native Qwen, 9.76%
for matched D and only 1.01% for mixed hard routing (within observed variation).
See [report](FA2_LOCAL_VERIFICATION_20261018.md). The local CUDA-compatible
torch 2.8 runtime differs from B200's pinned runtime; A100 cannot run the pinned
FA4 backward. No FA2-versus-FA4 B200 ranking follows from these measurements.
No production backend, environment specification, driver or remote command was
changed. Keep the B200 install directive in place until the user requests a check.

## Isolated FA2 runtime request — 2026-10-19

The user authorized a separate `train_fa2` environment. Its standalone spec
preserves the production torch 2.14.1/HF pins and requests a matching FA2
2.8.3.post2 CUDA 13 compiled wheel through `#i envs/train_fa2.txt +a` only.
The production `train` and `eval` specs are unchanged. The required wheel is
not an upstream release asset; runner-store availability and installation are
unverified. A missing-wheel result requires an operator-supplied matching build,
not a silent torch downgrade or a mismatched wheel. See
[environment notes](FA2_B200_ENVIRONMENT_20261019.md). No GPU work or Dropbox
monitoring is part of this installation request.

On 2026-10-19 the user restored ordinary Dropbox checks for 6Y3O; the earlier
deferral applied only while waiting for a replacement share. The first retrieved
status confirms installer success for `train` and `eval` at `e2dabb0`, but the
isolated `train_fa2` request at `960bbc9` failed with exit 3. Retrieve the detailed
installer log before attributing that failure or modifying any environment.

Read-only inspection `5f06832` on 2026-10-19 at 23:05 UTC verifies replacement
host `thiennh-p6-6y3o-worker-0`, eight B200s and driver 580.167.08. Both production
environments match the requested top-level pins and import the core libraries
and FA4 4.0.0b33. The partial `train_fa2` runtime has the same shared torch/HF
pins and successful core imports, but no flash-attn package. Its installer
failure cause is not returned by the status summary; wheel availability remains
unconfirmed. No environment/driver changes or GPU workload changes were made
during inspection. Eight active GPU processes were observed but not identified
as burns. This is package/import evidence, not CUDA kernel or training evidence.
See [receipt](results/b200_6y3o_environment_20261019.json) for exact versions,
GPU queries and downloaded-log hashes. Both commands.sh copies return to #0.

Follow-up `d3434c5` at 23:18 UTC diagnoses the FA2 failure from the node's
`/mnt/local/_env_status/train_fa2.status.json`: the requested FA2 CUDA 13 /
torch 2.14 / cp311 wheel is the sole `gap`, with `result: missing` and overall
`incomplete`; other packages succeeded. No matching wheel is in the inspected
node cache or the official release assets checked that evening. A compatible
operator-supplied build is needed before retrying this installation. The request
used the expected wheel name without establishing availability. This was a
prebuilt dependency lookup failure, not a failed GPU numerical test. No
environment/driver or workload changes were made. See
[diagnostic receipt](results/fa2_install_failure_20261019.json). The detailed
diagnosis supersedes the earlier unknown-cause note; commands.sh returns to #0.

The user then identified the PyPI source archive and emphasized node stability.
Source compilation is an alternative to the prebuilt-wheel request; the earlier
wording that only an operator-supplied wheel could resolve this was too strict.
The verified PyPI archive bundles CUTLASS headers and has an SM100 build target;
the GitHub automatic source archive omits the CUTLASS submodule. No B200 source
download/build was submitted. Any future attempt stays isolated, with bounded
compiler parallelism and verified resources/toolchain; preserve production
environments, system CUDA and drivers.

Fresh read-only check `161d804` at 23:27 UTC found all eight GPUs responsive,
same UUIDs/driver/PID mappings, zero volatile uncorrectable ECC and cgroup OOM
counters, about 1.83 TiB host RAM available and 24.1 TiB disk free. Production
versions match the preceding snapshot and core/FA4 imports pass. GPU PIDs
500–507 belong to burn launcher 433 (`/tmp/llm_pretrain_burn.py`, system Python
3.10), independent of the project environments. No process or environment
changes were made. See [health receipt](results/b200_6y3o_health_20261019.json).
These checks show no sign of damage, without claiming a full hardware audit
or pre-install file comparison. commands.sh returns to #0; no FA2 retry queued.


## Completed isolated FA2 source installation — 2026-10-20

The user authorized a compatible installation from official GitHub/PyPI inputs.
Controller #i installed CUDA 13 compiler/header packages only in train_fa2;
#d fetched the official PyPI 2.8.3.post2 sdist with a verified SHA256. No matching
upstream prebuilt wheel existed for torch 2.14.1. The initial source build failed
because FA2 selected C++17 while torch headers require C++20. Retry `2f8b97e`
changed only build-standard flags and compiled unchanged kernels for SM100.
`c11e34a` added an environment-local libcudart.so -> libcudart.so.13 link.
These fixes are preserved in `scripts/install_fa2_offline.py`.

The offline wheel installed successfully in 27 minutes; independent check
`d09dc58` completed 00:14:41 UTC. FA2 and compiled fixed/varlen forward/backward
interfaces import; existing package metadata is unchanged. Production train/eval
FA4 imports pass. All eight original burn workers remain, driver/GPU identities
match, and volatile uncorrectable ECC plus cgroup OOM/kill counts are zero.
See [receipt](results/fa2_install_success_20261020.json) and
[build notes](FA2_B200_ENVIRONMENT_20261019.md). commands.sh returns to #0.

This verifies installation/imports only. No B200 FA2 numerical, training,
throughput or resume test ran. GPU validation remains necessary before use;
production still uses its existing attention configuration. The env text lists
build prerequisites, not the now-nonexistent prebuilt-wheel request.

## Native baseline runtime and input contract — 2026-10-20

The user authorized the complete runtime/data/verification/baseline sequence on
6Y3O. Isolated stock FA2 2.8.3.post2 with torch 2.8.0/CUDA12.8 passes B200
numerics, but native eight-GPU update timing is not better than production FA4.
FA4 remains selected in train; production environments and NVIDIA driver are
unchanged. Full results: `results/attention_comparison_20261020.json` and
`ATTENTION_BASELINE_LAUNCH_20261020.md`. All nine numerical checks and twelve
timing jobs passed; stock FA2 is available separately, not adopted globally.

English input: nht10/cx_sampled at dd1b15134dc133a26d04f7edf45d9fe8315f4f52,
subset qwen3_0.6b_base_en_30B, 3,008 files/~135GB. Trusted per-file hashes,
metadata hashes and tokenizer revision are in resources/cx_english_20261020.json.
Eight disjoint controller download groups preserve the exact dataset layout.
Per-task controller progress lines may lag; actual node file counts advanced.

Commit c52d603 launches a bounded input wait followed automatically by all-file
hash/document checks, ordinary HF CPU cache fill, Accelerate copy/env check,
verified burn reclamation, eight-rank resume comparison, real smoke25/resume50,
and fresh native12 baseline5000. CPU preprocessing uses 128 workers, fixed
in the data recipe. Inputs and validation split remain unchanged. The supervisor
restores verified communicating burns after the GPU queue succeeds or fails.
The node acknowledged the waiting stage; this is not yet a training-start claim.

## Live baseline verified — 2026-10-20 04:35 UTC

The authorized setup/comparison/download/verification sequence is complete;
the fresh 5,000-update native Qwen3 12-layer baseline is actively training.
Execution root: `/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a02`,
output `prepared/baseline`, supervisor session `native12_train_20261020_a02`.
Implementation/launch commit caea83d; subsequent commits only retrieve status.
**Do not launch it again.** Source/execution commands.sh are returned to #0.

FA4/train selected after stock torch2.8/FA2 and compiler-fixed FA2 comparisons.
All 3,008 English dataset files and six tokenizer/config files pass pinned
SHA256 checks. Caches reused in 8.917 seconds with 29,278,633 train and 9,704
validation sequences. Original train/validation split, EOS, within-document
causality and reset positions are verified. Protected train/eval environments
and the NVIDIA driver remained unchanged.

Corrected eight-rank resume parity passed (weights max abs 3.7253e-9,
optimizer 2.3283e-10; scheduler/eight-rank RNG match). Real-model smoke25 and
resume50 both passed with matching training/data receipts. The first a01 smoke
failed before training due solely to the modern-cuDNN metadata getter; its
outputs remain separate. Only that reader changed; no precision flags changed.

At 04:35:32 UTC, baseline had logged step 340 (loss ~6.228), checkpoint250
complete including optimizer, scheduler, all eight RNG files, and matching
recipe. Validation250: loss 6.60917368, PPL 741.869745 over 9,915,698 scored
tokens/9,704 sequences. GPUs 0–7 each have one training worker, 97–99% utilization,
54,110 MiB device usage; guard held by the supervisor. No run failure observed.

Recipe: seed0, seq1024, micro16 × accumulation4 ×8 GPUs =524,288 input tokens,
full schedule57,220/cutoff5,000/warmup1,145, BF16, tf32False, checkpointingFalse,
AdamW3e-4 cosine floor.1, eval/save250, retain all checkpoints. The supervisor
will restore and verify communicating burns automatically after success/failure.
Estimated completion around 05:30 UTC (roughly one hour total, including eval/
saving; estimate from early observed progress, not a completion promise).

Receipt: `docs/results/native_baseline_launch_20261020.json`; report:
`docs/ATTENTION_BASELINE_LAUNCH_20261020.md`. Raw local evidence in
`temp/attention_study_20261020/`, especially baseline_health01.log and
retry_smoke25.json/retry_smoke50.json. Next status check should read the existing
queue, final baseline-5000.json, supervisor and burn-verified receipts; require
step5000/artifact success and live verified burns before declaring completion.

## Baseline completed — verified 2026-10-20 09:40 UTC

Native Qwen3 12-layer FA4 baseline finished **5,000 updates successfully** at
05:22:42 UTC. Runtime was 3,040.22 seconds in Trainer (50m40s), 3,083.11 seconds
for the complete baseline job (51m23s). Full schedule remains 57,220 steps;
all 20 checkpoints, every 250 steps through 5,000, are retained. The fresh
final-artifact check passed: weights, optimizer, scheduler, all-rank RNG files,
recipe and step metadata. Final held-out loss **3.46775378**, PPL **32.06463718**,
over 9,915,698 scored tokens in 9,704 sequences. Mean training loss 4.22683868
is the entire-run average, not the final validation metric.

Supervisor restored communicating burns by 05:23:48 UTC. Live check at 09:40 UTC
matched all eight workers 104144–104151 to the verified burn launcher, confirmed
the guard was released, and observed collective cycles advance 17,640→17,660.
No new training/evaluation was launched. commands.sh is returned to #0.
Final checkpoint: `/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a02/prepared/baseline/checkpoint-5000`.
Evidence: `docs/results/native_baseline_final_20261020.json` and local
`temp/attention_study_20261020/final_status01.log` (read-only inspection e19e34d).
The authorized baseline workflow is complete; further method training needs
its own scope from the user. Historical launch observations follow below.


## Dispatch optimization validated — 2026-10-20

Launch a5f65aa on 6Y3O completed at 16:45:53 UTC. Old code was frozen before
editing. Accepted changes in bdot/model.py remove redundant route discovery,
hard-ST zero filtering, range-discovery counts, repeated reference scatters,
duplicate RoPE work and eager diagnostic transfers. Home execution retains its
original expert autograd path; the earlier direct shortcut failed multi-update
GPU parity and was removed. No tolerance was relaxed.

All 32 CUDA old/new cases (SDPA/FA4, reference/grouped, eight routing modes) have
zero gradient and final-weight differences over three AdamW updates. Two-rank
checkpointed DDP checks pass; eight-rank A/B resume has exact restored/final
model, optimizer and scheduler state, exact data/RNG/routes and zero loss error.
Resume fixtures are reduced-width; this does not replace full-width pilot gates.

Two-repeat, all-eight-GPU full-update timing (FA4, seq1024, micro16×acc4):
D 0.707021→0.664589s, hard grouped 0.772241→0.733153s, soft grouped
0.999200→0.956848s. Native 0.617564s. Time reductions 6.0%/5.1%/4.2%; hard still
18.7% slower than native. Synchronizations per profile update: D 224→28,
hard 260→96. Synthetic timings exclude data preparation, eval and saves.
Grouped remains optional; default reference and strict source resume guards
remain unchanged. No environment, driver, data or research checkpoints changed.

Automatic burn handoff passed at 16:47 UTC. Independent 18:25 UTC inspection matches
workers 125740–125747, released ownership guard and collective cycles 6840→6850.
All 14 exported files are local at temp/dispatch_opt_20261020/results/; summary
SHA256 matches both remote inspection and completion manifest. Report:
BDOT_DISPATCH_OPTIMIZATION_20261020.md; durable machine-readable receipt:
results/bdot_dispatch_optimization_20261020.json. No research pilot launched.

Final full local CPU B-DOT suite: 48 tests passed in 321.35 seconds.

## 21 October 2026: pilot checkpoint diagnostics

See [Step 1 report](BDOT_PILOT_DIAGNOSTICS_20261021.md) and its JSON/CSV evidence.
Both learned P1 arms meet the response doc's weak-balancing rule. Default-route
interventions hurt their adapted checkpoints; trained R remains slightly better
than P1-A, and N-A still misses D + 0.01. Route/window/gate and balance replay
checks find no implementation defect. Saturated full-LM finite differences are
precision-limited; stable local checks plus one-router-at-a-time conditioned
full-LM probes pass without changing trained weights or relaxing tolerances.
The failed all-router-centered diagnostic is preserved. No new training ran.
P1-A′ (`argmax(softmax(z)+beta)`) is the indicated next candidate, not authorization
to launch it. Final burns/guard verified at 12:53:47 SGT; keep commands #0.

Do not infer historical mean logsumexp from logged mean(logsumexp²), or claim
an exact per-update beta audit when only periodic loads/checkpoint beta exist.
Use compact result-only archives with per-file hashes for many exports; the
latest directory export returned no valid paths. Sequential Dropbox downloads
avoid the observed parallel-download HTTP 429 limit.

## 21 October 2026: approved v3 batch

The user approved `BDOT_v3_pilot_plan.md`, three additional controlled ablations,
and delegated remaining operational choices. V5 adds both placement FiLM maps
to V4; V6/V7 retain only the query/MLP-input map of V2. These exploratory arms
are excluded from the predeclared gate candidate pool. The agreed count is
12 v3 jobs plus P1-A′ to close v2, 13 training jobs total. The required seed-1
D/V2/C2 runs change model/routing seeds only; data seed/order remains zero.

Scientific recipe stays the complete 4,768-update / 96-warmup pilot, FA4,
sequence 1,024, microbatch 16 × accumulation 4 × eight GPUs. Retain steps
1,000/2,500/4,768 plus the latest complete optimizer checkpoint, pruning only the new experiment's older
checkpoints after all-rank RNG/optimizer/recipe files are present. Existing
v2 outputs and shared caches are preserved. Source/options must stay fixed
while a feature is in use; numerical/code failures halt for review.

Default options reproduce v2. RF caps the full router logit and selects on
masked `p + beta`; PF maps only allowed off-home module/step pairs. Its vectors
start at zero without consuming RNG, use main LR and no decay. The tanh cap
bounds candidate gaps, but does not guarantee nonzero parameter gradients.
Missing conditional controls/second seeds must be reported as pending, never
silently treated as a successful two-seed gate. The current queue contains the
agreed 13 jobs; conditional follow-ups require applying the plan to their results.

Acceptance passed at 16:18:15 SGT; the research queue started immediately.
B200 numerical/grouped tests, exact eight-rank V3/V5 resume, T8 and all four
v2 checkpoint carry-over checks passed. At 16:23:27 SGT, P1-A′ had reached
step 390 with a complete checkpoint-250 and unchanged source hashes. The remaining
runs/evaluations are automatic; burns restore through the existing supervisor.
See `BDOT_V3_IMPLEMENTATION_20261021.md` and
`results/bdot_v3_launch_20261021.json` for precise evidence, scope and limitations.


## V3 batch completed — 22 October 2026

All 13 runs/evaluations finished at 05:32:12 SGT without retries; final states
and milestone retention verified. No new arm beats D. V6/V1 are effectively
tied near loss 3.5437 versus D seed 0 at 3.494464. V2 loses to D by 0.063505
and to random C2 by 0.002109 on the two-seed mean. P1-A′ has loss spikes and
ends at 5.859132. PF and the wider window did not improve quality in this pilot.
The automatic V1 selection triggers the missing conditional C1 control; its
formal two-seed gate is pending, not a pass. V1 misses two known quality
criteria, so it does not qualify for near-miss second-seed confirmation.
No conditional follow-up was launched during the status check.

Burns restored at 05:33:23 and independently verified at 05:50:33: known workers
on all eight GPUs, collective progress and released guard. Status 4f48b80 and
export ba4d79b; all 124 archived files hash-verified locally. See
`BDOT_V3_RESULTS_20261022.md` and `results/bdot_v3_final_20261022.json`.
