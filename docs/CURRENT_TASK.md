# Current task

## Active: A2 Phase 0 implementation; no B200 launch

User authorized implementation after review of [A2_pilot_spec.md](A2_pilot_spec.md)
v2.1 and [A2_novelty_and_feasibility.md](A2_novelty_and_feasibility.md).
Implementation guide: [A2_IMPLEMENTATION.md](A2_IMPLEMENTATION.md).

Added `efficient_llm/a2/`: fixed 127-run matrix, original-layer pruning with
pinned main OpenSci code, pure full-vocabulary KL, native HF Trainer/Accelerate,
exact update-indexed LR, deterministic corpus-wide reservation and EOS packing,
separate tuning/final sets, BF16 export/reload, residual/bypass/Gaussian probes,
phase queue/retry rules, decision reports and plots. Token counts are reused
from the checksum-verified published `document_index.parquet`; no long-document
filter or repeated Qwen tokenization of the reserved billion-token pool.

Local checks use `temp/a2_test_env` (Transformers4.57.1 overlay, original train
runtime untouched), native pinned OpenSci source with tiny models, and a meta
shape check for the full 1.7B parameter count. Unit and two-process CPU evidence
is archived at `docs/results/a2_implementation/validation.json`. These are NOT
released-weight CUDA/BF16/throughput results. Test inputs/results remain in
ignored `temp/a2_*`; no full model weights or dataset were downloaded this turn.

Next: acquire/verify actual pinned public assets using the authorized controller,
prepare the fixed data, record secondary acquisition/skip decisions, run all
actual-parent B200 preflight stages, aggregate/review T1–T13/T15 and measure the
whole workload cost. Only then construct the execution configuration for the
127-run pilot under the existing supervisor, with Accelerate cache verification,
known-burn reclamation/free-GPU checks and verified automatic burn handoff.
The heal entry point refuses missing/mismatched preflight and baseline evidence.
No deferred Phase1, B200 install, GPU action, execution push or new timer occurred.
`commands.sh` remains #0; old research outputs preserved and user design/review documents unchanged.

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
