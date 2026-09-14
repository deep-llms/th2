# Seed-29 remaining offline replication — 2026-09-14

## Scope and policy

User authorized an automatic remaining seed-29 workflow, and explicitly chose:
**apply the same Depth-Delta inclusion thresholds separately to seed 29**.
This is a recorded inclusion-policy amendment from the earlier
`seed17_then_all` implementation, before inspecting seed-29 memory results.
It does not change the thresholds, primary hypotheses, model, table capacity,
reader, optimizer settings, corpus, data-order policy, or token budgets.
Seed-17 results and its Delta exclusion are not retroactively changed.

No common retraining, seed43, D_val, online writes, compiler acceleration,
architecture tuning, cleanup or automatic resume/retry is authorized here.

## Single-owner queue

`commands.sh` calls `bash scripts/launch_seed29_replication.sh` once.
That activates train_env and starts a persistent tmux session:
`ccm_replication_seed29_20260914_a01`. Inside it,
`scripts/pilot_seed29_replication.py` owns every transition:

1. Revalidate prepared files/coverage and the existing seed29 common checkpoint.
   Keep current all-eight original burns active during CPU preflight.
2. Verify A3 success/fresh heartbeat; create A3's STOP_IDLE_WATCH and verify its
   exact observer exits voluntarily. Stop only verified burn-worker pidfds.
3. Wait 30s/free check, then another 30s/free check.
4. Compile all five tables from seed29 theta_4B on GPU0 using the unchanged
   reference compiler: microbatch4, isolated batch256, full 1B compile tokens.
   Original communicating burns occupy GPUs1–7.
5. Verify tables/counts/provenance/shuffle. Reclaim only that burn group and
   repeat all-eight free checks.
6. Train Stage1 sequentially on all eight GPUs:
   Contextual → Isolated → Shuffled → Shallow → Delta → Grad.
   Each gets 977 updates / 256114688 adaptation input tokens.
   Verify each checkpoint, optimizer, unchanged backbone/table and paired reader.
7. Evaluate unadapted Base plus all six arms on full D_dev; GPU0 evaluates
   sequentially while GPUs1–7 burn. Verify every evaluation.
8. Restore all-eight burns for CPU comparisons and the seed29 Delta decision.
   Stage1 complete marker is written only after its panel passes.
9. Stop the same owned all-eight burn group; two 30s/free checks. Train Stage2:
   Base → Contextual → Isolated → Shuffled → Grad → optional Delta.
   Each starts from the original seed29 common model, fresh optimizer and paired
   reader; Grad also gets a fresh table. No Stage1 weights are carried over.
   Each gets 3815 updates / 1000079360 continuation input tokens.
10. Evaluate every scheduled Stage2 model on D_dev and verify all results.
    Restore all-eight original burns, then compute paired comparisons and
    validate the final panel. Only then publish workflow complete.json.
11. Maintain a 60-second idle-burn health observer until explicitly disarmed.

Training uses the proven **torchrun, eight ranks**, not Accelerate; no
Accelerate config is needed. BF16, activation checkpointing, microbatch8,
loss chunk1024, save every1000/final, same exact schedules as seed17.
Single-GPU evaluation uses microbatch8/loss chunk1024 and one shared
seed29 Contextual diagnostic table for all bin definitions.

## Delta gate and reporting

Use `ccm delta-decision --replication-policy per_seed --cluster doc_id`.
The decision binds seed29, its corpus/vocabulary and its common writer hash.
All conditions are required:

- upper95(Delta − Contextual hit NLL) < 0;
- upper95(Delta − Shuffled hit NLL) < 0;
- upper95(Delta − Contextual miss NLL) ≤ 0.002;
- Delta overall NLL ≤ Contextual overall NLL.

The same 10,000 paired document bootstrap replicates and seed20260913 are used.
A failed scientific inclusion gate omits Delta, not the five required arms.
Missing/corrupt artifacts, failed training, or failed validation stop the queue.

Stage1 reports Contextual versus Isolated/Shuffled. Stage2 reports Contextual
versus Isolated/Shuffled/Base/Grad, and, if admitted, Delta versus
Contextual/Isolated/Shuffled. These are individual seed29 D_dev results, not
the final three-backbone decision. No automatic seed43 launch.

## Paths and provenance

- Code: /mnt/local/deep-llms_th2
- Conda: /mnt/local/conda-py311/envs/train_env
- Common: /mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed29_20260913_a01/common/checkpoint-15259
- Common SHA256: `da85b43b1509f0f8b5e807f951b624126f17d673edd505b8f041d8413f104038`
- Fresh output: /mnt/local/_outputs/deep-llms_th2/ccm_replication_seed29_20260914_a01
- Handoff log: same path with `.handoff.log` appended.
- Inputs: /mnt/local/_data/deep-llms_th2/ccm/pilot_v1_20260913_a01
- Subdirectories: tables/, stage1/{train,eval,reports}/, stage2/{train,eval,reports}/.
- Delta decision: stage1/delta_decision.json; heartbeat: status.json.
- Terminal event: `seed29_replication_verified_and_burns_active`.

The original common/data core hash remains
`055f0518853e76487a4e2f31b81f1441113d4fceab322b5e211359a7d58f2fa9`.
New policy-support core hash:
`bbbd831476fd942d1131754ad3b57fd0323ed7f41aa03b8cb783bea497f8a2ff`.
The new validators distinguish immutable old provenance from the reviewed
new launch code; no old artifact is rewritten to claim a new hash.
Only CLI/Delta decision guards changed inside ccm/. All model, compiler,
training-update, data and evaluation math is unchanged. Three trailing blank
lines were removed from canonical copies to match the execution source bytes.

Old run-specific scripts with the old exact core guard are historical; do not
launch them unchanged against the new core. Future final-val handling of any
different optional Delta panels must be reviewed when that stage is authorized.

## Safety and recovery

One Python owner executes and waits for each child, with per-minute read-only
status. No separate stage watcher can race it by restarting burns.
No name-based signals, process-group kills, PID1 kills, checkpoint deletion or
automatic retries. Failure preserves logs; burns are restored only after
owned children have exited and GPU ownership/free checks permit it.
If a child/unknown workload remains, it is preserved for inspection.

Before a subsequent authorized GPU job, disarm this workflow through:
`/mnt/local/_outputs/deep-llms_th2/ccm_replication_seed29_20260914_a01/STOP_IDLE_WATCH`.
Verify exact observer exit before stopping verified burn workers.

## Verification and monitoring

68 existing/new CPU tests passed, plus the tiny seed29 common → compile →
Stage1 Delta → Stage2 Delta test (69 total). It verifies fresh Stage2 reader
initialization and rejection of another seed's inclusion decision.
Mocked include/exclude queue paths and failure/ownership paths passed.
This is CPU correctness evidence, not a claim that the full run has finished.

### Verified remote startup

Launch commit `2c035fa`; read-only startup export `28562d1`.
Remote input gate passed, including original seed29 model/optimizer hashes,
immutable corpus/vocabulary, and current reviewed core hash.
A3 observer exited cooperatively; verified burn workers were stopped; both
30-second GPU-free checks passed. Compiler started at 21:22:07 UTC Sep14.
Retrieved heartbeat at 21:24:09 UTC: compiler PID 68604 on GPU 0, original burn
workers 68078–68084 on GPUs 1–7, each at 98% utilization, with seven connected
NCCL ranks. No error in the retrieved handoff. Empty compiler log at startup
is expected: this reference compiler does not print per-batch progress.
Stage1/Stage2 are queued and have not yet been verified complete.
Evidence: canonical `temp/seed29_replication_live_a01/`.

Monitor by fresh #2 exports, never repush the #1 launch. Pull status.json,
handoff log, current stage log and validation JSONs. AWS/runner-system errors
must be reported to the operator, not fixed through retries/kills/cleanup.
Results are retrieved through controller #2 and dev Dropbox, never outbound
network connections from B200.

Based on seed17: compilation ~4h57m, six Stage1 training bodies ~1h39m,
five Stage2 training bodies ~4h52m, plus evaluation/checks. Rough queue estimate
**12–14 hours**, with about one extra hour if Delta enters Stage2; not a guarantee.
