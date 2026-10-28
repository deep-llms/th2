# Remaining Phase 0A preparation — 28 October 2026

The current NEM S1 queue keeps its frozen source and recipe. The new code is
`scripts/a2_remaining.py`; no numerical model, loss, Trainer, data order,
optimizer, selection rule, or GPU supervisor was changed.

Implemented and CPU tested:

- Import reviewed, SHA-pinned result/baseline/terminal receipts. Reject changed
  files, incomplete queues/burn handoffs, duplicate IDs, nonfinite metrics,
  incorrect eight-rank sample order, wrong LR schedules and failed save/reload.
  Historical provenance remains intact. An original baseline may predate a
  coordinator-only commit; runtime/data/spec identity and its loss must match.
- Select each parent's LR from its own complete three-point tuning grid. After
  all three S1 pairs succeed, subtract their exact 30 run IDs from the registered
  matrix: 28 fresh fits remain (18 short S2/S3/S4, six main S2, two NEM long S1,
  and two T14 repeats). No automatic retry import; an unexpected historical
  failed/retried fit requires explicit handling rather than silent retraining.
- Identify missing S72/S188 surveys. Never freeze S2 while a branch that passes
  pinned T3 identity is unmeasured. Exclude only T1/T3 failures, never diagnostic
  loss flags. Require both valid annealed parents. Reuse the original S2 function.
- Build the existing runner's prerequisite manifest: up to six eight-GPU branch
  surveys, CPU set selection, then one representative numerical job covering the
  union of frozen patterns. It has no heal or throughput jobs. CPU jobs omit the
  `gpus` key, as required by the proven runner.

Validation: all 58 A2 tests pass, including eight new tests. A local replay of
20 actual archived DCLM/FWE fit receipts passes all import checks. Test logs:
`temp/a2_remaining_tests_20261028_final.log`; actual-receipt replay helper and
local-only lock: `temp/a2_replay_history.py`, `temp/a2_known_history_local.json`.
The full 30-fit plan is tested with synthetic records; NEM final receipts do not
exist locally yet. Do not describe NEM or S2 as completed.

Before GPU execution, still required:

1. Retrieve NEM's successful final artifacts and independently verify burns.
2. Build/review a remote-path history lock from actual receipts, including the
   original DCLM/P300 standalone survey; pin all actual artifact hashes.
3. The prerequisite manifest is now connected to the existing native launcher
   (`--stage remaining-prereqs --history LOCK`). It verifies branch weights/code,
   current native provenance, cumulative worst-case timeouts plus cleanup reserve,
   then reuses Accelerate copying, fresh burn identities/free-GPU checks and
   native automatic burn restoration. Actual GPU execution awaits NEM handoff.
4. After S2/numerical checks, create the current-provenance preflight/baselines
   for remaining heals and use the exact pending list. The old generic queue
   still runs its full matrix; do not call it unchanged and repeat S1/grid fits.

The helper prepares/validates work, not a complete new training launcher.
Existing historical numerical byte/AST locks remain enforced. No new GPU work
was submitted in this turn, only a read-only health request.

At approximately 12:59 SGT, the runner reported an SSH hostname-resolution failure
for health submission8920863. All remote writes are frozen pending operator/system
recovery. This does not establish whether NEM training stopped. Keep outputs and
workloads intact; local code/tests may continue, and the existing watcher may
check read-only receipts. Do not resubmit commands as an infrastructure repair.

Recovery addendum, 13:26 SGT: the same pending health8920863 subsequently ran
successfully (snapshot13:08, receipt13:11); no retry was submitted here. NEM
continued through five grid fits. Remote-write freeze is lifted on this evidence.
The new launch gate has CPU tests for exact required assets, completed-history
dependencies and budget rejection. Its first actual GPU prerequisite stage is
still pending; numerical core and the running NEM frozen source are unchanged.

Validation after launcher integration: 59 A2 tests pass; actual20-receipt replay
still passes. Log: temp/a2_prerequisite_tests_20261028.log. No GPU execution
claim for this new stage. Numerical byte/AST locks still pass.

Local continuation preparation, 14:00 SGT: scripts/a2_remaining_heals.py now
implements exact pending-fit execution after a current all-pattern gate/baselines.
It subtracts the30 prior fits, keeps each dataset/parent's selected LR, preserves
T14 permutation suffix through the single allowed half-LR retry, rejects infra
failures, checks every successful result's coverage/schedule/baseline/roundtrip,
and reserves600seconds for handoff before the Phase0A cap. It does not acquire
GPUs or launch by import. Four new CPU tests exercise all28 pending runs, explicit
subsets, retry identity, caps, non-divergence failures and malformed receipts.
Actual full-pattern gate/baseline assembly and launcher connection for these heals
remain pending; the prerequisite stage is separately connected and ready to
validate real history. Do not describe this executor as already running.

Final preparation:64A2tests pass and all30actualcompletedfits+6annealedsurveys
replay successfully. The original DCLM/P300survey predates the launch-scope
amendment/data-spec metadata field: recognize only its exact reviewed artifact
SHA, verify both historical byte/ASTlocks, and check the approved spec transition
without rewriting originalprovenance. Changed survey values are rejected by test.
Node-path history lock: resources/a2_completed_s1_20261028.json.

14:34 SGT integration: `scripts/a2_heal_stage.py` verifies reviewed successful
prerequisite hashes, rechecks BI selection, requires identical numerical
code_files/runtime/data for check reuse, preserves the original check in its
envelope, verifies historical T8/recipe evidence, scores six native all-pattern
baselines, then invokes only the28 pending fits. It is connected as launcher
`--stage remaining-heals --prerequisites LOCK` (plus existing history/prior/grid).
No old matrix/grid fits are repeated. Worst-case original-fit timeouts plus
baselines/cleanup must fit remaining18h Phase0A cap; each retry is separately
budget checked. T15 diagnostics are recorded and never change eligibility.
70 CPU tests pass, including numerical-provenance/tampered-receipt rejection,
all-six-baseline CLI ordering, registered28-run execution and retry/T14 checks.
Actual prerequisite final receipt and GPU execution remain pending; current
branch job keeps its frozen source602fbce. Log temp/a2_heal_stage_tests.log.


At14:37SGT the first branch stage had six successful surveys but failed in CPU
S2 assembly: the exact original DCLM/P300 survey predates parent_sanity metadata.
The fix accepts that omission ONLY for its original locked SHA; pinned history
already requires a successful parent pre-baseline, and healing requires it again.
It never marks arbitrary missing sanity metadata as passed. Actual30fit/12survey
replay now selects S2=[1,2,3] DCLM/FWE, [1,2,5] NEM. All71CPUtests pass.
Original failed supervisor and successful survey receipts are retained under
results/a2_branches_20261028/final_surveys. Recovery reuses all six surveys through
resources/a2_completed_surveys_20261028.json and executes only CPU freeze + GPU
numericalcheck. No numericalcore or scientific selection rule changed.

Recovery a02 was refused before GPU acquisition: the inherited successful-job
handoff helper requires complete.json. Added an explicit --failed-handoff receipt
lock for this exact failed CPU freeze after six completed surveys. It verifies
terminal cleanup and restored communicating burns from SHA-pinned artifacts;
the native supervisor still rechecks live PID/start identities and GPU freedom.
This does not turn failure into success or relax scientific gates. A separate
CPU test uses the actual archived failure receipts and rejects changed/foreign
paths and incomplete burns. Next recovery uses a fresh a03 output.

Further review before healing fixed finite tuning divergence handling: native
heal may return status=complete, diverged=true. The pending coordinator now
executes the one half-LR retry instead of treating this as infrastructure failure;
a second divergent result remains incomplete and does not count as successful.
A test exercises two finite divergences then continuation to the next fit.
All73CPUtests pass. Native training/loss/optimizer code unchanged.

Supplementary downstream preparation (not launched): the old KV adapter's
checkpoint20752/layout constraints and native BF16 casting must not be copied
unchanged. Reuse benchmark_data's offline tasks and likelihood batching, but
load original OpenSci parents through a2.model.load_parent and pruned students
through reload_student, verify kept-layer maps/weights/source hashes, and match
A2 scoring's FP32 parameters plus BF16 autocast. Use the already-pinned GPT-NeoX
tokenizer (EOS0), plain causal positions/no chat template/no generation. Preserve
paired prompts, examples and seed, record truncation; choose one fixed context
within4096 for every compared checkpoint. Existing suite assumes1024 and exactly
two KV models, so a separate native-model entry point/metadata collector is needed.
Before any GPU evaluation, confirm offline nine-task files and lm-eval runtime on
node; no installs or downloads are authorized as an implicit compatibility fix.
Prioritize the remaining registered healing stage and do not overlap its GPUs.

Local report preparation at16:33SGT: temp/a2_aggregate_remaining.py maps only
receipt paths to the local archive, compares every other history-lock field,
then reuses load_history, verify_result and results.report. Actual30historical
plus3new receipts replay successfully; duplicate IDs, modified LR and modified
data provenance are rejected. Partial data remains INCOMPLETE; unexecutedPhase0B
is explicitly absent, not a failed experiment. This changes no deployed source.
Final reporting still requires all expected original IDs/retries, terminal
scientific results and independent supervisor/burn verification.

Native downstream adapter prepared locally in eval/a2_harness.py. It wraps an
already verified model from load_parent/reload_student, using FP32 parameters
and CUDA BF16 autocast, EOS0, native SDPA, fixed<=4096context and likelihood only.
It reuses the original harness request handling; the Qwen adapter is unchanged.
CPU checks on the pinned tiny native OpenSci model cover independent likelihood,
padding/batching, left truncation, physical-prune save/reload, and rejected
precision/EOS/generation settings. The lm-eval0.4.13 logits reuse cache gives
incorrect scores when identical input tokens have different context/continuation
boundaries; the regression fixture detects this, and A2 disables logits_cache.
No GPU evaluation has run. The suite/checkpoint identity collector and real
full-model GPU parity remain required; confirm offline node benchmark files and
compatible harness runtime before scheduling. No training numerical code changed.

At17:27SGT the node confirms all nine existing offline benchmark snapshots pass
manifest hashes. a2_env has torch2.14.1/transformers4.57.1/datasets4.8.5/tokenizers
0.22.1, but no lm-eval. Do not change the live environment or load native A2
weights using the old eval environment's incompatible Transformers.

Prepared eval/a2_likelihood.py: a native-only batched token-request scorer, no
lm-eval dependency, FP32parameters/BF16CUDAAMP, full-vocabulary likelihood,
canonical left truncation/rightpadding/EOS0 prefix and original request order.
It matches the pinned harness on CPU, including the cache-boundary regression.
Public text requests with the actual SHA-verified GPT-NeoX tokenizer also match
independent native logits, including emptycontext/trailingwhitespace. Six tests
pass; no actual-weight GPU evaluation yet. Training/numericalcore is untouched.
Next evaluation preparation: confirm the existing eval_env runtime, prepare
canonical task/token requests in that CPU environment, score those requests
using a2_env, then replay scores to canonical task aggregation. This avoids
installing/importing libraries across environments. Request identity/checkpoint
collector, CLI/orchestration and actual GPU parity remain required before launch.

Prepared eval/a2_request_bridge.py, CPU-only TemplateLM capture/replay. Capture
raises RequestsCaptured after canonical tokenization and before metrics; it never
publishes placeholder scores. Replay requires identical text/token request order,
matching request hash, complete finite nonpositive likelihoods and boolean greedy
flags. ThreeCPU tests pass, including an actual offline ConfigurableTask roundtrip
through canonical task aggregation; final metrics/prompts equal direct HFLM on
the tiny native OpenSci model. Native training and scoring precision are unchanged.
Still missing: persisted request/scoring CLI, checkpoint identity and supervised
suite, plus actual-weight GPU parity after the training handoff.

The17:57 read-only runtime probe queried envs/eval_env and reported that path
missing. Historical successful downstream command uses envs/eval/bin/python3.11
(temp/kv_downstream_health_command.sh), so absence of the actual eval environment
is NOT established. Next read-only check must use that exact historical path.
No package installation, environment repair or cross-environment import occurred.

18:31SGT actual runtime check confirms the existing envs/eval interpreter:
torch2.14.1, transformers5.9.0, lm-eval0.4.13, datasets4.8.5, tokenizers0.22.2.
It can host CPU request preparation/aggregation; native model loading stays
in a2_env with Transformers4.57.1. No package/env changes or cross-env imports.

Prepared eval/a2_requests.py prepare/replay CLI: local tokenizer/data checksums,
fixed canonical zero-shot task settings and seed42, persisted request/protocol
hashes, score-to-input binding, exact task/sample counts and exclusive output.
Capture aborts before metrics; replay uses canonical aggregation with actual
scores. Five CPU tests pass, including full JSON prepare/native-score/replay
on an offline task, matching prior direct-harness tests, tamper/overwrite rejection.
Raw test log temp/a2_request_cli_tests_final.log; checksum in status_1831 archive.
Native checkpoint identity/scoring CLI, suite and full-weight GPU parity remain
before downstream deployment. This is local preparation, not an executed eval.

19:01SGT local-only preparation: eval/a2_checkpoint.py collects actual parent or
healed export file hashes from reviewed input/result/reference/frozen-set locks.
It verifies exact completed-fit recipe/LR/order/roundtrip, original layer mapping,
matching native code, and file identity before/after native FP32 reload.
This records current export bytes; it does not invent a training-time weight hash.
The coordinator must provide the reviewed locks and wait for terminal handoff.

eval/a2_score.py scores a SHA-pinned prepared request artifact using the unchanged
native scorer and load_parent/reload_student paths in a2_env. Output binds request,
protocol, checkpoint hashes and runtime; no lm-eval dependency, no GPU ownership
operations, no environment changes. Three CPU tests passed on tiny24-layer native
model export→18-layer prune/reload, batched-vs-single scoring, and changed
weight/code/layer/receipt/request rejection. Not actual1.7B/B200 acceptance.
Still needed: reviewed checkpoint identity-plan builder, supervised paired suite,
and actual-weight GPU parity/smoke after current training/burn handoff.

19:32SGT local-only downstream preparation: eval/a2_suite.py plans native scoring
on eight explicit GPU lanes, serializes multiple checkpoints assigned to the
same GPU, and loads each checkpoint once for the entire shared request set.
The caller must first validate training/burn handoff and numerical gates; this
module neither acquires GPUs nor launches anything on import.
Collector checks all nine tasks, exact sample counts/prompts across models,
request/checkpoint/precision/runtime identity and score-file hash used by replay.
ThreeCPU tests pass, including mocked ninejobs/eightlanes without GPU execution,
paired accuracy aggregation and changed/missing artifacts/prompts rejection.
Need reviewed checkpoint selection/identity-plan construction, supervised launch
integration and actual full-model GPU parity before deployment. No new GPUjob.
