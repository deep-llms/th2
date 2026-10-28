# Current task

## Latest: bounded DCLM/P300 S1 grid authorized — 28 October 2026 SGT

User approved dependency-based launching instead of waiting for all parents.
Preparing `launch_a2_preflight --stage early-grid`: actual representative checks
for S1, unhealed baseline, then exactly three 0.03B LR-grid fits (114 updates
apiece) and only the registered half-LR retries. No P1T/S2/other dataset launch.
The existing supervisor retains Accelerate verification, exact burn ownership
reclaim/free checks, guard and automatic communicating-burn handoff.
Target root: /mnt/local/_outputs/deep-llms_th2/6y3o-a2-early-grid-20261028-a01.
Session a2_early_grid_20261028_a01. Prior/source/data unchanged; original T1/T8
receipts reused via explicit checksum and unchanged numerical-code locks.
Spec's appended dependency amendment governs this bounded launch. No changes to
loss, optimizer, schedule, dataset/order or scientific selection. The gate can
only authorize DCLM/P300 S1's registered grid and retries; no full-pilot claim.
Estimated 25–35 minutes after queue start, plus ~7 minutes per retry; one-hour
hard cap. Charge all elapsed time to Phase 0A alongside prior 322seconds. Full
pilot workload report remains pending; report progress/ETA in Singapore time.
Submission032f258 pushed/remote-head verified at08:29SGT, frozen source4c2c385.
Launch acknowledged in Dropbox00:32:41UTC /08:32:41SGT. Actual parent/input
hashes, historical-reuse contract and copied8GPU/BF16 Accelerate config passed.
Raw logtemp/a2_grid_launch01.log SHA
3c39413a31b42fd176fac0c9ce1a71835897fc3ee4bf6d99c0f857e876bdba44.
Supervisor submitted; live numerical/training stage still needs health receipt. Expected
completion09:00–09:10SGT if it starts promptly and needs no retry. Pending health
command: temp/a2_early_grid_health.sh. Do not repush the executable launch.

## Latest: lean preflight implemented — 28 October 2026 SGT

User approved running shared implementation checks on one actual representative,
not all twelve parents. The appended amendment in A2_pilot_spec.md governs scope.
`check` now requires DCLM/P300 and covers every distinct frozen S1–S4 pattern
across all three datasets, including different S2 choices. One post-S2 job;
no repeated T8, no automatic healing. Its receipt is mandatory in the heal gate.
Per-parent load/hash/config/finite-forward/RoPE checks and weight-dependent
FP32/BF16, BI, Gaussian and loss measurements remain. Native source identity and
structural configuration are verified before sharing code-test evidence.

The old prepared data is accepted only for this exact test-scope amendment;
original manifest/order/receipts remain untouched and both spec hashes are
recorded. Main model, loss, optimizer, data and supervisor are unchanged.
Local CPU tests passed (see A2_IMPLEMENTATION.md); no new B200 job submitted in
this scope-change turn. commands.sh remains #0. Actual shared numerical checks
are still pending, not claimed passed. Remaining scientific surveys, frozen S2,
shared checks and whole-workload report still precede any healing. Initial T8
and DCLM/P300 survey remain complete as recorded below; preserve their source
identities when reconciling the final preflight report. No resampling is needed.

## Initial actual preflight COMPLETE — 28 October 2026 SGT

Fresh healthc3fe89a at07:46:31SGT confirms successful terminal artifacts:
T8completed07:39:03; DCLM/P300 survey completed07:43:24 in261.18seconds;
whole two-job queue322seconds. Supervisor burn handoff verified07:44:30 and
independently still healthy07:46:31:8workers638429–638436, guard released,
advancing collective cycles140→150 /155.45→166.55GiB per rank. Reverify identities
before any later reclaim. New predecessor burn script is
`/mnt/local/_outputs/deep-llms_th2/6y3o-a2-initial-preflight-20261028-a01/source/resources/llm_pretrain_burn.py`.

Actual parent T1precision passes: BF16-AMP small loss2.723300243 vs fullFP32
2.723201842, gap0.000098401nats; final2.737793032. Other-theta loss2.844251513.
Position-bucket loss not monotonically decreasing is a diagnostic flag only,
never a parent exclusion. No scientific healing arm launched. Remaining11parent
surveys, frozenS2 selection, numerical/removal/roundtrip gates, unhealed baselines
and whole-workload cost report still required. Initial launcher does NOT queue
those automatically. Measured representative survey4.35minutes suggests about
48minutes for11similar surveys alone; other checks/scoring/load/save add time.
Do not claim a complete all-parent preflight or a fixed pilot ETA from this.

SHA-verified final receipts: results/a2_preflight_20261028/initial_complete_0746.json
anddclm_P300_survey.json. Full local artifactstemp/a2_actual_health02; raw logSHA
138b6903444cac33cf9ce7d7b58c5cbbdc55d063c7eb1ea7037dfcfc109cdb37.
Controller download session reportedENDED at07:43; not yet independently verified
all54model files, so do not treat it as full asset acceptance. commands.sh#0.

## Actual B200 preflight running — 28 October 2026 SGT

User requests the real run. Execute the spec-required actual-parent preflight,
report its outcome/cost before any scientific healing. Initial launch bbf9997,
source20b61d0, acknowledged in07:36 launch log. Actual data and DCLM/P300 pinned
weight/code SHA checks passed, resource Accelerate cache copied and env verified:
8B200/BF16, a2_env Torch2.14.1+cu130 / Transformers4.57.1 / Accelerate1.13.0.
Root `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-initial-preflight-20261028-a01`;
session `a2_initial_preflight_20261028_a01`. First discarded12-update T8, then
DCLM/P300 survey; no heal queued. Existing supervisor owns burn reclaim/free
checks and automatic communicating-burn restoration on success/failure.
Launch logtemp/a2_actual_launch01.log SHA
abcf5ba8a2f0e4581f6a19cb34946f11e8db02c6c4b2fd61b2726bae1f74abb1.
Health17b37bb at07:41:05SGT verifies exact known burn workers stopped, all8GPUs
empty at07:38:01before launch, and12-update T8 successful at07:39:03. T8 measured
1.199023s/update /218,631tokens/s, after discarding first2startup timings;768samples
across8ranks exactly match intended order. Peak allocator44,738,788,864bytes on
rank0. End-to-end T8job60.77s includes loading/startup; throughput excludes those.
Survey running since07:39:03, all8workers638142–638149 active under native launch.
Supervisor637822 owns guard and automatic burn handoff; do not stop/relaunch it.
Nine artifact hashes and70source files verified. Rawtemp/a2_actual_health01.log SHA
9ca063da590f8014be957326fa149e653efe89518fdd388b0a07bac475d3f6af.
Receipts results/a2_preflight_20261028/health_0741.json andthroughput.json.
commands.sh restored#0; jobs run from frozen source. Next retrieve survey/terminal
supervisor/burn receipts, then complete remaining parent/numerical gates and cost
report before scientific healing. No full pilot ETA or acceptance claimed.

Read-only readiness863ab40 at07:32:43SGT confirmed50/54asset files SHA-verified:
only Nemotron S72/S188 still pending. Initial launcher checks only its actual
parent dependencies; unrelated downloads continue and every later parent still
requires pinned hashes. Three readiness/job-manifest tests passed after this
narrow launch-only change. Main model/loss/data/schedule/supervisor unchanged.
Readiness receipt: results/a2_preflight_20261028/readiness_0732.json.

## Latest preparation continuation — 28 October 2026 SGT

CPU preparation COMPLETE07:11:32SGT (790.30seconds), confirmed by read-only
healthd0c4797 at07:12:39. Original run/source unchanged:12exported artifact SHA
checks and all59 frozen source files independently verified against2589987.
Packed heal207,519×4096=849,997,824tokens; tuning1024×4096=4,194,304;
final2461×4096=10,080,256 from all11,717validation documents. Fixed reservation
1,200,002,619Qwen tokens, T7 document id/content disjointness passes. Preserve
all output; no rerun needed. CPU job exit0 plus terminal artifacts both pass.
Root `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs`.
Evidence: results/a2_preflight_20261028/input_status_0712.json; full small artifacts
in ignoredtemp/a2_inputs_health03. Raw log SHA
2ef837628b88dcf0cb59c6da18f32666d6c5ffd71a7fa82245314b2c61aa011c.

Weight download still ongoing:26/54files SHA-verified, including all4DCLM weights
and FWE/P300. FWE/P1T and later parents not ready. Do not redownload/restart; next
retrieve controller completion and verify remaining actual bytes. GPU burns still
owned by prior review supervisor; no GPU job or environment/driver change.
Fresh health receipt acknowledged; commands.sh returned#0 with code preparation.

Prepared `scripts/launch_a2_preflight.py`: only discarded12-update real T8 plus
one DCLM/P300 survey, both8GPUs, under unchanged train_then_burn supervisor.
Requires successful main preparation, exact spec/data/asset contracts, actual
packed bytes and required DCLM/P300 weight/code hashes BEFORE GPU reclaim.
Other parent downloads may continue; their own loaders still require exact hashes. Copies
resource Accelerate config and verifies accelerate env; supervisor owns exact
burn PID checks, free-GPU checks and automatic communicating burn restoration.
No heal/automatic pilot entry point in this initial-stage manifest.34CPUtests
pass, including readiness rejection and bounded-job manifest tests. Actual
preflight remains unlaunched while parent downloads complete; later surveys,
full numerical checks, full-workload estimate and prelaunch report remain.

## A2 actual preflight preparation — 28 October 2026 SGT

User asked to continue. Proceed with pinned checkpoint acquisition, frozen-data
preparation and actual-model preflight; report preflight before any healing.
No scientific healing run is queued. Active spec remains A2_pilot_spec.md v2.1.

Read-only inventory11efc8d observed06:42:03SGT: authorized6Y3O alive, old burn
workers634055–634062 under633986 unchanged; no other GPU workload. English
cx_sampled_dd1b1513 has all3008 files, matching sizes and pinned manifest SHA;
36,495,424train documents and11,717validation documents. Full shard SHA checks
subsequently passed before sampling (see latest health below). Existing Qwen tokenizer present. New A2 asset folder
absent; about22.6TB disk free. No environments/driver/outputs changed.
Raw inventory temp/a2_assets_inventory.log SHA256
705824080d73feee0b483fe490d78a5fd460dce5c0483a5c91fdf0deb1c77453.

Input lock resources/a2_inputs_20261028.json contains12 pinned parent weights,
main-source code/config/tokenizers (54 files), existing pool/tokenizer contracts.
Controller54-file download ab350b8 to /mnt/local/_models/deep-llms_th2/a2
acknowledged STARTED (ID2026-10-27_22-46-06), completion pending.
Do not consume before per-item success and actual SHA verification. CPU worker
scripts/a2_prepare_inputs.py verifies tokenizer/data inputs then prepares the preregistered
whole-pool random prefix with GPT-NeoX. It has no GPU/network/process control.
32 tests now pass, including corrupt inputs and streaming secondary formats; progress logging changes
no sampling or packing semantics. Prior GPU review evidence stays historical.

Secondary locks resources/a2_secondary_20261028.json: first fixed public shard
of DCLM-baseline, FineWeb-Edu v1.0.0 (the original release), Nemotron-CC actual
high-quality. Full Nemotron source read locally and SHA/ETag verified. Only these
three shards are requested via controller; stream the first256x4096tokens each.
No need to load/decompress the entire corpus or change the generic baseline.
T15 resource a2_logged_lr_20261028.json binds public archive and exact run names:
Nemotron188000 exactLR.004; other requested exact steps absent in the matching
log members. Nearest values are explicitly context only, never substituted.
CPU launcher scripts/launch_a2_inputs.py snapshots source and uses the existing
job runner, four-hour timeout, no GPU/burn/guard control. Launched3bd6cb8,
reviewed source2589987, at06:58:22SGT; health32f323c at07:01:15SGT confirms
running/no error, frozen source hashes match, all3secondary sets complete and
all751train+1validation English shards SHA-verified in136.37seconds. Nemotron
packed bytes match the independent dev probe exactly. Main tokenization follows;
no data completion marker yet. Root:
/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01
Data will be at run/inputs/data; assets/secondary_map.json under run/inputs.
Session a2_inputs_20261028_a01. Do not restart or overwrite this healthy job.
Receipt docs/results/a2_preflight_20261028/input_status_0701.json.
Latest user questioned necessity of secondary/tokenizer checks: explained the
spec's secondary corpus diagnostics versus mandatory tokenizer/count identity;
no additional optional test expansion. Keep focus on main data and actual model.
Secondary downloads all3complete; parent request remainsRUNNING. The controller
progress line can lag: actual06:58inspection found16/54files including a partial
S72weight. No failure inferred and no duplicate download/relaunch. Data can
prepare concurrently once its tokenizer bytes are present: it never loads parent
weights, and its receipt explicitly says model_weights_verified=false. Every
parent still requires actual weight SHA verification before GPU loading.
Local real Nemotron shard + pinned GPT-NeoX test produced exactly256x4096tokens;
receipt docs/results/a2_preflight_20261028/data_probe.json. No model scoring.
Detailed preflight record: docs/A2_PREFLIGHT_20261028.md.

## A2 review complete; full pilot not launched

Active design: [A2_pilot_spec.md](A2_pilot_spec.md) v2.1 and
[A2_novelty_and_feasibility.md](A2_novelty_and_feasibility.md).
Implementation: [A2_IMPLEMENTATION.md](A2_IMPLEMENTATION.md).
User requested a second correctness review and B200 testing where needed.
Receipt: [results/a2_review_20261028/validation.json](results/a2_review_20261028/validation.json).

Fixed final/secondary scoring errors incorrectly triggering lower-LR retries,
strict run/retry/provenance validation, secondary contrasts/report selection,
and per-branch T3 failures. Loss, architecture, stream order, scientific matrix
and schedules are unchanged. 30 local tests, two-process CPU equivalence and
11 unchanged supervisor tests pass. Public metadata for all12 pinned parents
passed identity checks; released weight bytes have NOT been downloaded/tested.

B200 smoke64789a3 completed **28October06:22:12SGT**; read-only health124e9f9.
One/eight GPU native OpenSci random fixture,12 updates each,64 sequences/update,
sequence64, width256, vocabulary1024,4 parent/2 student layers. Actual head-width
2048/vocabulary50304 BF16 checkpoint test has zero loss/gradient difference.
One/eight GPU max parameter difference1.4768448e-5 within the fixed tolerance;
exact sample order/LR, frozen teacher and uneven evaluation coverage pass.
BF16 save/reload loss difference0; a4096-token scoring probe also passes.
This is NOT actual-parent T1/T8 or acceptance of the1.7B scientific workload.

Isolated env `/mnt/local/conda-py311/envs/a2_env/bin/python3.11` installed:
Torch2.14.1+cu130, Transformers4.57.1, Accelerate1.13.0. Original train environment
remains Transformers5.9.0/Torch2.14.1; driver untouched. Pinned code-only files in
`/mnt/local/_models/deep-llms_th2/a2_code_review_20261028` verified before use.
Resource Accelerate config copied and8GPU/BF16 `accelerate env` verified;
only verified known burn workers stopped, all GPUs checked free before tests.

Automatic communicating burns verified **06:23:23SGT**, independently healthy
**06:23:45SGT**; guard released, progress50cycles/55.52GiB collective payload/rank.
Workers634055–634062 under633986 at that observation. Historical identities:
reverify before stopping. Burn source:
`/mnt/local/_outputs/deep-llms_th2/6y3o-a2-review-20261028-a01/source/resources/llm_pretrain_burn.py`.
Root `6y3o-a2-review-20261028-a01`; small artifacts locally SHA-verified, including
62 unchanged source files and all complete-marker artifacts. Raw receipt
`temp/a2_health01.log`, SHA256122549326d86745d3fb88e621ea793841a1ca76334339bc9aee645361bd62dad.
commands.sh restored#0; no healing or other GPU job queued, no new timer.

Next before any real pilot: acquire actual pinned checkpoints, prepare frozen
heal/tuning/final data, record secondary acquisition/skip decisions, run actual-
parent T1–T13/T15, measure full-workload T8 cost and send the prelaunch report.
Only then configure the127-run scientific queue under the existing supervisor.
No deferred phase is authorized by a decision label. Preserve old outputs.

## Complete: downstream screen of long-recovery A/B, 28 October SGT

User-requested evaluation complete; [results/protocol](KV_DOWNSTREAM_20261028.md).
Nine full English zero-shot tasks; native SDPA/BF16 weights, context1024, seed42,
26,618examples per arm. Eight-task mean: A60.6003%, uniform25B58.4273%, delta
−2.1729points; paired within-task example-bootstrap95%CI[−2.6646,−1.6825].
B lower on all9tasks. LAMBADA accuracy52.2802→50.2620%, PPL10.2766→11.6019.
No downstream parity or measured inference-efficiency claim. One training seed.

Successful fresh a02 launch dc7c0ba, health a850e76. Actual GPU native-logit and
manual-likelihood gate exact (zero errors); eight-example/task smoke passed;
full evaluation150.27s on8concurrent workers. All stages return0, full benchmark
counts/paired prompts and unchanged checkpoint hashes verified. Two likelihood
requests per arm truncated at1,024; others fit. Existing eval environment reused,
no installs, driver changes or model/training modifications. Resource Accelerate
config copied and8GPU/BF16 env verified; known burns stopped and8GPUs free.

Queue complete01:21:47SGT. Communicating burns restored/verified01:22:58SGT,
independent live check01:23:02SGT:8workers630628–630635, launcher630559,
source under this run's source/resources/llm_pretrain_burn.py; guard released,
20cycles/22.21GiB collective progress/rank. Historical identities; reverify later.
commands.sh restored#0. No more jobs queued; previous overnight timer stopped.

Root `/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-downstream-20261028-a02`.
Local archive `docs/results/kv_downstream_20261028/final/`, raw
`temp/kv_downstream_health01.log` SHA256
`48a696f89032f4b4d6fbf8e5a8488fcf8df7fa85b35f73715048ef835c865782`.
All exported artifacts SHA-checked; complete-marker hashes verified. Full
per-example outputs remain onB200 with local byte/SHA receipts. All checkpoints
preserved. Initial a01 failed in tokenizer manifest validation before GPU stop or
output creation; fixed dict-versus-digest comparison and regression-tested.

## Complete: 1.36B-token dense/uniform25 pair, 28 October 2026 SGT

Project objective [PROJECT_IDEA.md](PROJECT_IDEA.md); recipe
[KV_LONG_RECOVERY_20261027.md](KV_LONG_RECOVERY_20261027.md); results
[KV_LONG_RESULTS_20261028.md](KV_LONG_RESULTS_20261028.md).
Both authorized arms completed20,752 steps /1,360,003,072 tokens each. Final
BF16 PPL: dense A17.657243, uniform25 B18.200775 (+3.078%). Paired-document95%CI
for B−A NLL[0.028418,0.032295] nats/token, passing the prespecified5% screening
threshold. B still has worse PPL. Reused256-document prefix probe, one seed;
no downstream equivalence or measured deployment speed claim.

Queue finished00:38:45.992 SGT28October. Automatic communicating burns restored
and verified00:39:51.837; independent live inspection00:41:30.782 shows8 known
burn workers628303–628310 under launcher628234, guard released, progress to130
cycles/144.35GiB logical collective payload per rank. Historical PID identities
only: reverify before any future authorized stop. No more jobs authorized/queued.
Old overnight timer remains stopped. commands.sh restored#0.

All six jobs returned zero. Data/model hashes, original shared dense A-512 parent,
actual per-rank sample order, exact once-only1,328,128-document coverage and
save/reload checks passed; both parent/reload max NLL differences zero. Local
recomputation reproduces summary.json exactly. All23 saved full checkpoints
per arm remain, with optimizer/scheduler and8rank RNG files; last step20,752.

Accelerate cache copy and8GPU/BF16 verification, exact burn reclaim/free-GPU
check, uniform25 numerical gate, and both8step8GPU smokes passed before fits.
Training runtime A82.25min, B79.40min; queue including checks/setup2h46m.
CPU preparation~2h48 beforehand. User correction recorded in PROJECT_NOTES:
do not blindly scale diagnostic data filtering into larger training jobs.

Run `/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-long-20261027-a01`,
launch1c968cf, final read-only health85e22e0. All18 exported artifacts SHA-verified
and archived`docs/results/kv_long_20261027/final/`; local`temp/kv_long_health03/`.
Raw receipt`temp/kv_long_health03.log`, SHA256
 a32aa5a6b72563ea3e04b972f245abbe461bf26aaf9455550f826777eeb97e51.
Large model/optimizer checkpoints remain on B200; do not remove them.
Earlier progress archive`docs/results/kv_long_20261027/status_2244/` remains.

## Completed: bounded KV-sharing recovery research, 27 October 2026 SGT

Project objective: [PROJECT_IDEA.md](PROJECT_IDEA.md). Authorized overnight scope:
[KV_RECOVERY_OVERNIGHT_20261027.md](KV_RECOVERY_OVERNIGHT_20261027.md).
Final report: [KV_RECOVERY_FINAL_RESULTS_20261027.md](KV_RECOVERY_FINAL_RESULTS_20261027.md).
Workspace /disk/thuat/efficient_llm. Report times in Singapore time (UTC+8).

All four additional training jobs in that earlier authorization are complete.
Final matched BF16 PPL: dense A 19.2416, uniform25 20.4610, upper25 20.6189,
full50 22.6557, uniform25 Q/K/V-only 23.0779. All sharing variants fail the
upper-95%-CI <= log(1.05) gate. Stop this short conversion/recovery recipe;
no pruning sweep or speed-only evaluation. One model/seed, reused 256-document
prefix probe: do not generalize this to all KV sharing or downstream accuracy.

### Final QKV job and evidence

Run `/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-qkv-20261027-a01`,
session `efficient_kv_qkv_20261027_a01`, launch3932ad4; final healthd1e92f3.
Receipt confirms Accelerate cache copy and eight-GPU BF16 verification. Smoke
8steps/8GPUs and full2048-step fit pass. Same original dense A-512 parent,
134,217,728 tokens, exact sample order and recipe as dense/all-parameter controls.
Frozen bytes unchanged; selected Q/K/V weights changed; parent/reload max NLL
difference zero. Local recomputed summary equals node summary exactly.

Queue completed **09:36:37.676 SGT**. Communicating burn handoff verified
**09:37:48.530 SGT**. Independent live health **09:55:46.983 SGT** confirms eight
known burn workers, guard released, advancing collectives through 1,250 cycles
and 1,387.94 GiB logical payload/rank. No environment/driver/model/data change.

13 exact exported artifacts SHA-verified, archive
`docs/results/kv_qkv_20261027/`; local `temp/kv_qkv_results_20261027/`.
Raw health `temp/kv_qkv_health01.log`, SHA256
`3ffd72cea3ebd77556ac10ea603a412dd10732fe3eaa9e98abe75e7f8db51e32`.
Large checkpoints remain on B200. Prior outputs and caches preserved.

Latest burn source is this QKV run's `source/resources/llm_pretrain_burn.py`,
launcher620091, workers620160–620167. These are historical identities, never a
substitute for fresh PID/start verification before any future authorized stop.
No active training or evaluation remains. commands.sh restored to **#0**.

Existing timer `temp/kv_recovery_watch_20261027/STOP` is set after verified final
results/handoff. No replacement timer/agent. Do not restart research from stale
queued wake prompts; completed cap is4/4. New authorization is recorded above.

### Earlier recovery stages

Recovery a02/a1847c0 complete07:39:16SGT: denseA19.2416, oldB22.3924 (+16.37%).
B starts from already-shared B-512 and is not a matched placement initialization.
Three placement jobs bca23aa complete08:29:36SGT from SAME denseA-512:
uniform25 +6.34%, upper25 +7.16%, full50 +17.74%. All artifacts/data/reload and
burn checks pass. Reports KV_RECOVERY_RESULTS_20261027.md and
KV_PLACEMENT_RESULTS_20261027.md; SHA archives under docs/results/.

Node source copy omits .git; source identity uses SHA manifests. Earliest recovery
a01 failed before Python/GPU because of a git metadata shell header; corrected
a02 preserved all outputs. Delayed Dropbox receipts are not run failures.

## Completed initial pilot (historical)

User approved a bounded pretrained KV-sharing × MLP-pruning interaction pilot.
All six fresh 512-step arms completed successfully on original 28-layer
Qwen3-0.6B-Base. Total 201,326,592 recovery tokens. Native HF Trainer/Accelerate,
native SDPA, identical actual sample order and teacher-distillation recipe.
Final training-validation PPL: A 20.03790, B 26.22461, C 21.67554, D 28.69731,
E 30.77849, F 32.49516. Recalibrating after sharing did not help this recipe.
This is a 256-document × 1024-target prefix probe, not full-corpus/downstream PPL.

Training source `7abbd37`, run `6y3o-efficient-kv-mlp-20261027-a03` completed
**05:26:25 SGT, 27 October**. Burns restored/verified 05:27:36; independent
health `3ba782c` at 05:28:28 confirmed advancing eight-rank burns and released
guard. Compact result bundle exists under this run's `result_export/`.
Its manifest SHA256: `57d09d8d87d4794401ac3e2c317850f6f77132ea45a381cbff83537f70048d02`.

Evaluation-only precision check `c140449` completed **05:39:00 SGT** in 34.67s:
`6y3o-efficient-kv-mlp-inference-20261027-a01`. BF16-weight PPL A/B/C/D/E/F:
20.03850 / 26.22722 / 21.67890 / 28.70302 / 30.78658 / 32.50281, matching the
profile precision with unchanged ranking. Same probe and trained model code
verified. Accelerate copied/verified, eight-GPU BF16 env confirmed.
Burns restored 05:40:11; independent health `381f1da` at **05:42:22 SGT** verified
advancing burns and guard released. No environment/driver changes.

**Archival complete:** exact-file export `8809832` returned 29 files, all
SHA256-verified locally against node receipts (folder export `59e30f9` failed
without changing results). Full local archive `temp/kv_mlp_results_20261027/`;
small durable artifacts `docs/results/kv_mlp_20261027/`.
[Final report](KV_MLP_RESULTS_20261027.md). Model/optimizer checkpoints remain
on temporary B200 storage; they were not pulled through the small-file channel.

Step-512 interaction D−B−C+A is +0.011546 nats [0.006544,0.018057]. E−D is
+0.070013; F−E +0.054275. Post-recovery single-layer pruning sensitivity
correlation A/B: Pearson .985, Spearman .922. B8/8k eager decode: A 18.390 ms,
D 15.821 ms (paired speedup 1.162×), but D's inference PPL is 43.2% higher.
Shared arms store exactly half the KV. No useful quality/cost joint benefit;
retain as a negative pilot rather than scaling this recipe. One short recovery,
model and seed; no 8k-quality or downstream claim. commands.sh #0.
**Historical pilot complete. Superseded by the recovery authorization above.**

## Verification and historical failures

Eight CPU model/training tests and 55 infrastructure checks pass. Actual-weight
GPU gate and eight-GPU smoke passed. Every arm passes exact one-pass data
coverage, matched per-rank order, frozen teacher and save/reload checks (max
checked NLL difference zero). Checkpoints require `load_compressed`, not vanilla
AutoModel; consumer KV retains gradients; decoder checkpointing is rejected.

Attempts a01/a02 stopped in the gate before training; outputs preserved. Corrected
reference QK precision and separated full-model same-kernel BF16 checks from
fixed-input cross-kernel checks. FP32 full explicit-math output/gradient errors
4.53e-5/9.91e-5; same-SDPA independent BF16 decoder output error zero; fixed-input
BF16 output/gradient maxima .001873/.012473. Full-model explicit-math BF16 errors
.399/.681 remain documented; cross-kernel interchangeability is **not accepted**.
See `docs/results/kv_mlp_20261027/gpu_gate.json` and the design document.

## Runner and latest ownership

Remote `git@github-share:deep-llms/th2.git`, main; this worktree serves development
and execution. Node `thiennh-p6-6y3o-worker-0`, eight B200s. Code
`/mnt/local/deep-llms_th2`; all outputs under `/mnt/local/_outputs/deep-llms_th2`.
Do not repeat migration/cleanup; never repush unchanged executable commands.sh.

Python `/mnt/local/conda-py311/envs/train/bin/python3.11`: torch 2.14.1+cu130,
transformers 5.9.0, accelerate 1.13.0, datasets 4.8.5. Model
`/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6`; data
`/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513`.
Contract `resources/cx_english_20261020.json`; config `resources/accelerate_b200.yaml`.
No GPU-node outbound network; controller #d/#i inputs and #2 exports only.
Dropbox label 6Y3O; credentials/map under ignored DOT/temp. Never print contents.

Latest verified burn ownership, **09:55:46 SGT**: workers620160–620167,
launcher620091; source QKV a01/source/resources/llm_pretrain_burn.py.
Guard released and collective progress verified. Reinspect exact identities
before any future authorized stop. Rules in GPU_SAFETY.md.

## Earlier evidence

[WINDOW_COMPACT_RESULTS_20261027.md](WINDOW_COMPACT_RESULTS_20261027.md),
[SELECTIVE_DECODE_RESULTS_20261026.md](SELECTIVE_DECODE_RESULTS_20261026.md),
[CONTEXT_BUDGET_RESULTS_20261026.md](CONTEXT_BUDGET_RESULTS_20261026.md),
[ROUTER_PILOT_RESULTS_20261026.md](ROUTER_PILOT_RESULTS_20261026.md), and
[JOINT_GROUP_RESULTS_20261026.md](JOINT_GROUP_RESULTS_20261026.md).

Prior learned group/context routing has not demonstrated useful whole-model
speed over the fastest dense control or cache-capacity savings. Compact kernel
is opt-in; native SDPA default. Removing sinks from the frozen half-local policy
severely worsens PPL. Do not restart those queues automatically. Raw archives
remain under ignored temp; durable reports/results under docs/. Large training
checkpoints remain on temporary B200 storage; small metrics are backed up.
