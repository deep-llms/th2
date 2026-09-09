# Project notes

Status (2026-09-09): switched to the six-layer Qwen3 English pilot. The user
requested reuse of sparse-embedding's sampled text and training workflow.
All six5000-step training arms and the42-checkpoint evaluation sweep finished;
fine-tuning is underway. Complete raw evaluation results are retained locally
under `artifacts/eval_42ckpt_20260909_a01/results/` (see CURRENT_TASK.md for
source checksums and provenance). No scientific comparison is recorded yet.
GPT-2 preprocessing was stopped and is superseded.
Current source/commands are documented in CAPACITY_EXPERIMENTS.md.

## Destination eval/finetune smoke — 2026-09-09

Full-suite inputs are now verified. Controller#d executiona5b3177 downloaded
the five previously missing repositories into the retained benchmark root
`/mnt/local/_data/deep-llms_th2/benchmarks/hf`. Executionfbfeb3b verified all133
reference hashes, all78 English eval splits, HellaSwag/ARC-Easy/XNLI training
splits, and a74-task two-example CPU smoke for tiny B0/C. These remain input/
correctness checks, not research scores. Terminal reports were pulled via8792aff
and matched to source hashes; see CURRENT_TASK.md. All42 requested checkpoints
are retained, all eight GPUs were free, and no existing data was deleted.

Authorized sweep: all six arms at250/500/1000/2000/3000/4000/5000, full English
PPL and78 benchmark results per checkpoint, then independent English task
fine-tunes with seeds42/123/456. Keep one GPU per job (eight concurrent), not
eight-GPU DDP with a changed global batch. The checked-in shell handoff gates
the entire84-job evaluation stage before the378-job fine-tuning stage and
preserves original checkpoints. New outputs are separate from pretraining.

Correction/retest: user approved fixing the benchmark precision issue described
below. Commitbbcce4e explicitly configures HFLM BF16 forwards and FP32 softmax,
without changing pretraining or fine-tuning updates. All85 local regressions
passed. Execution5665bf6 then passed15 destination integration tests,12 CUDA
precision cases and all six checkpoint5000 smoke workers. All18 task/checkpoint
cases observedBF16 before/after fine-tuning; master weights/softmax remainFP32.
PPL-path, two-update fine-tuning, save/reload and source-weight/config hash checks
passed. After the final30s wait all eight GPUs were free; no burns restarted.
Corrected summary: `temp/remote_logs/swt_smoke_fixed_final_20260909_summary.json`,
SHA256 `bf89a99c724622fa125f694921b8073e8d63d6af59a067d8dc5996376b5c04f7`,
matched against the hash printed on B200. The failed pre-fix smoke below remains
historical evidence, not the status of the corrected code. No full research
evaluation or finetuning sweep was run, and new English benchmarks still need
their separate download/verification step.

All six training arms finished at01:49:11 UTC. The post-training burn verifier
timed out, but the independent burn was live on all eight GPUs at07:09 UTC,
with advancing communication counters. At the user's request, the verified
eight burn workers were stopped (only worker PIDs, not PID1/launcher/group).
An initial system-Python import failure sent no signals; the corrected command
activated `swt` and verified the helper path before stopping. All GPUs were free.

Installed isolated `swt_eval` through controller#i; left `swt` unchanged. The
destination smoke used all six actual checkpoint5000 models on GPUs0–5:
2048-token synthetic PPL, real HellaSwag/ARC-Easy/XNLI subsets, two BF16 optimizer
updates/task, and save/reload. All passed those checks, including unchanged
original weight/config hashes. Fourteen destination integration tests also passed.

The overall smoke correctly FAILED: all six tiny CUDA precision probes and
all18 actual checkpoint/task tests observedFP32 benchmark calls despite requesting
BF16. This is the known nested lm-eval autocast issue, not a pretraining defect;
fine-tuning optimizer updates useBF16 correctly. The wrapper is not fixed yet.
No full evaluation/finetuning sweep or burn restart was launched. All eight GPUs
were verified free after the smoke workers exited and a30s wait.
Evidence: `temp/remote_logs/swt_smoke_final_20260909_summary.json` (SHA256
`cee1315c89700b93a3324afdbe9149e03c12ee7f4c9da86fed539a023464cec4`) and
six per-arm reports. Synthetic/subsample smoke metrics are NOT research scores.
See CURRENT_TASK.md for commits, paths, environment and the pending fix decision.

## Batch-size benchmark completed — 2026-09-08

Current launch supersedes the stopped10k runs below: user-authorized5k screening
of all six arms was submitted in execution `594b3ac` and started at18:09:41 UTC.
Output root: `/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_5k_s42_20260908_a01`.
SWT_STOP_AT_STEP=5000 controls training and all verification gates consistently;
effective batch512 gives5,242,880,000 input tokens per arm. No max_steps override:
the full English pool and one-epoch LR schedule are unchanged. All GPUs were
verified free, Accelerate copied/verified, and eight destination handoff tests
passed. Cache regeneration was active (~2%) in the18:11 UTC snapshot. The queue
will train B0/A128/A256/A512/C/D sequentially, verify checkpoint5000 for every arm,
then launch persistent communicating burns only after success and free-GPU checks.

Latest state: at the user's request, a03 was cancelled and cleaned at17:58:34 UTC.
Only its exact experiment queue and eight verified GPU workers were stopped;
a03 outputs/checkpoints and qwen_en_map160_batch1000 were deleted. Sampled data,
tokenizer/model and unrelated HF caches were preserved. Stagewise directories
have no cache-/tmp- leftovers. All eight GPUs are free; no job/burn was restarted.
The discussion of5k screening has not yet resulted in a5k launch or code change.
See CURRENT_TASK.md for the verified cleanup log and current authority.

The user subsequently authorized a fresh rerun of all six arms. Execution
`f528935` started the `qwen6_allarms_10k_s42_20260908_a03` workflow at17:22 UTC;
all GPUs were verified free, Accelerate copied/validated, seven destination
handoff tests passed. Cache regeneration was approximately10% in the17:25 UTC
snapshot. It will use the unchanged batch16/accumulation4 protocol, stopping
each arm at10k with the full-epoch schedule, then verify all results before
launching the persistent communicating burn. See CURRENT_TASK.md for evidence.

Subsequent user-authorized cleanup completed at16:54:52 UTC: the stopped
`qwen6_allarms_10k_s42_20260908_a02` output (including checkpoint6250),
`batch_benchmark_20260908_a01` output and dedicated `qwen_en_map160_batch1000`
cache were deleted on B200. About519 GiB of directory usage was removed.
Sampled data, tokenizer/model and unrelated HF benchmark cache remain intact.
All eight GPUs were free at completion; no new run was launched. Local results
below are preserved, but previous statements about retained remote checkpoints
are historical. See CURRENT_TASK.md for exact paths and cleanup evidence.

All 12 eight-GPU profiles passed, finishing at15:44:49 UTC. Each used five
warm-up and20 measured optimizer updates, actual English training cache,
BF16 and2048-token sequences. Effective batch stayed512:16×4×8 versus32×2×8.
Downloaded JSON verified all eight ranks, finite timings and identical data
fingerprints. Times below are median optimizer-update seconds, excluding
periodic evaluation and checkpoint saving; this was not a full training run.

| Arm | Batch16 / accumulation4 | Batch32 / accumulation2 |
|---|---:|---:|
| B0 | 0.7070 | 0.7010 |
| A128 | 0.7253 | 0.7202 |
| A256 | 0.7091 | 0.7014 |
| A512 | 0.6861 | 0.6728 |
| C | 0.5584 | 0.5543 |
| D | 0.6950 | 0.6890 |

Batch32 fits every arm, but the measured throughput gain is only0.7–2%,
potentially within short-test variability. Peak reserved memory increases
from75.9–82.8 GiB to150.0–163.2 GiB per GPU. No material end-to-end speedup
is established. All GPUs were free at completion; research training remains
stopped with B0/checkpoint-6250 retained. No automatic restart or burns.

Evidence: `temp/remote_logs/swt_batch_progress_20260908_1629.log` and
`temp/remote_logs/swt_benchmark_complete_20260908_1632.json` (SHA256
`cf29dbe1115db58a86c7f9f01b1cf6e55913b97b18cc5822472fb94868e4e7c7`).

## Purpose and success criteria

Test the allocation ideas in §12 of the historical design with a Qwen3 backbone:
random-initialized stock B0/A128 first, A256/A512 controls, then C/D T1
stagewise width schedules. See CAPACITY_EXPERIMENTS.md for current code and usage.
The document remains a hypothesis/test plan, not evidence of architectural gains.

## Infrastructure

Development: `nguyenhuuthuat09/stagewise_widening_transformer`, local
`/disk/thuat/stagewise_widening_transformer`, branch `main`, remote `origin`.
Execution: `deep-llms/th2`, branch `main`, remote `runner` in development and
`second` in `/disk/thuat/th2_runner_clean_probe`. See `GIT_PUSH.md`.

th2 remains one 8×B200 node, `thiennh-p6-oish-worker-0`. Runner substitution
`@PROJECT@` is **deep-llms_th2**, not the development repository name.
Code lives at `/mnt/local/deep-llms_th2`; external data/models/outputs use
`/mnt/local/_{data,models,outputs}/deep-llms_th2`.

Before migration, old source files were recoverably moved to
`/mnt/local/_project_archives/deep-llms_th2/before_stagewise_20260907_a01`.
Read-only verification commit `30b25dd` confirmed all 23 entries absent from
the project folder and present in the archive, plus hashes of the two main
training files. Only `commands.sh`, `_run_log_`, `_previous_run_status.log`,
and `_dl_active.txt` remained in the code folder. The runner does not propagate
Git file deletions automatically (verified by the preceding commands-only push).
Preserve runner logs/state during future code-folder cleanup.

Existing data, checkpoint and environment directories were retained. Their
presence does not validate the new English/GPT-2 corpus or B200 ML environment.
Dropbox remains the same th2 folder (historical label `h100-1`); credentials and
private URLs stay in ignored local files and are never part of publication.

## Reproducibility

Record input revisions/manifests, preprocessing, seeds, configuration, resource
budget, metrics and validation criteria. For ML projects, distinguish training
from held-out results, document batch/precision/scheduler settings, and retain
checkpoint/evaluation provenance. Do not claim speed from tiny smoke tests.

## First Qwen training authorization — 2026-09-08

The user authorized correctness fixes, safe burn reclaim, training, completion
validation, and communicating-burn restart. The user explicitly selected
B0 -> A128 -> A256 -> A512 -> C -> D. Batch16 x
accumulation4 x 8 B200, BF16 SDPA, seed42, English-only existing sampled text,
2048 context, 10k-step cutoff within a full-epoch LR schedule; see the frozen
settings and workflow in CAPACITY_EXPERIMENTS.md. Preparing new HF map caches
while existing burns remain active avoids GPU idleness during CPU preprocessing.
This is normal trainer tokenization/packing, not replacement corpus sampling.

Read-only preflight137fb5d confirmed the same node, working swt imports,
35 saved English training shards/36,595,514 documents, 11,822 eval documents,
local Qwen3 tokenizer directory, and all eight known burn workers. The live
/tmp burn SHA256 matches the reviewed resources copy. No cleaning needed.

Before launch, fixed two review findings: reject incomplete optimizer/RNG
resume state, and reject nonzero label smoothing that HF mis-shifts for the
custom model class. Also record/compare world size and effective batch on
resume. Default CE behavior and architecture weights are unchanged by these
guards. Handoff verifies weights and checkpoint state, not PID disappearance.

Local final verification:51 tests passed, plus the all-six-arm two-rank BF16
CPU smoke. Development64472ea was pushed; execution50340a8 was rejected
before execution by a runner false positive:
`BLOCKED | GUARDRAIL: outbound push pattern(s): train.py:126: if training_args.push_to_hub:`.
The flagged line rejected Hub uploads. No GPU stop, cache preparation or training
occurred. The user subsequently instructed removal of this unnecessary guard
to follow the proven sparse-embedding implementation. Retain HF's default of
disabled Hub uploads and never enable uploads in launch settings. The direct
outbound-upload prohibition and this example are now in local AGENT_GUIDE.md.
That correction was committed as 792ea47. On the user's subsequent re-run
request, execution commit **1226141** submitted a fresh a02 pipeline. At
2026-09-08 13:25:50 UTC, the runner log confirmed source/host/ownership checks,
all 51 destination CPU tests passing, and CPU tokenization-cache preparation
underway while existing burns remained untouched. GPU smoke/training had not
yet been observed. Run root:
`/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a02`.
Evidence: `temp/remote_logs/swt_retry_start_20260908_a02.log`.
No cleanup or replacement corpus sampling was performed. See CURRENT_TASK.md
for the exact next gates and required completion artifacts.

## English evaluation and fine-tuning implementation — 2026-09-08

Added separate evaluation/fine-tuning tools with `--languages en` defaults;
the active pretraining code and frozen20-file launch manifest are unchanged.
`eval/eval_parallel.py` queues one PPL and one benchmark worker per checkpoint.
`finetune/run_all.py` queues independent task/checkpoint/seed workers. Both
use explicitly selected free GPUs, unique outputs, full result-coverage checks,
and nonzero failure propagation; neither stops workloads or starts burns.

Official lm-eval0.4.10 definitions were inspected at upstream tag commit
f7d0b116 and reused for both task prompts and multiple-choice scoring. Local
dataset paths are applied to task configs in memory, not shared package YAML.
English zero-shot tasks are XNLI, Belebele, XStoryCloze, PAWS-X, HellaSwag and
ARC-Easy; ARC-Easy adds the previously missing zero-shot fine-tune comparator.
Future languages are explicit and unsupported pairs cannot become English
fallbacks. Separate PPL per language and target-weighted combined NLL avoid
cross-language packing/averaging errors. See EVALUATION.md for full protocol.

Fine-tune defaults retain3 epochs/LR2e-5/tasks and3 seeds, but use FP32 master
weights with BF16 autocast and HFLM-compatible continuation tokenization. This
is not exactly the old BF16-master/separately-tokenized fine-tune protocol;
record these differences when comparing historical results. English-only
defaults still imply54 fine-tunes for6 checkpoints ×3 tasks ×3 seeds.
There is no scheduled evaluation job or B200 environment change in this work.

Final verification:62 tests passed in109.606s, including actual lm_eval0.4.10
offline fixture scoring on all six model types, tiny BF16 CPU fine-tuning,
save/eval round trip, token-boundary consistency, language selection and queue
failure contracts. Log: temp/evaluation_final_full_suite.log. Optional harness
dependencies were installed only in the isolated dev temp/evaluation_test_env;
the existing swt environment was not changed. These are correctness tests,
not real benchmark results or B200 throughput evidence.

## Qwen3 migration — 2026-09-08

The user chose Qwen3-0.6B with six layers and the existing Qwen-sampled English
corpus. Stock B0 has 249,969,152 unique parameters. Custom models retain the
interface/width experiment arms but now use Qwen3 GQA (2:1 query/KV ratio),
Q/K normalization, FFN width 3d and head dimension 128. See the current guide
for recounted arm budgets; historical Llama parameter counts do not apply.

train.py now follows the old HfArgumentParser/TrainingArguments/Trainer
workflow, loading old saved text and using batched multiprocess Dataset.map.
The serial SQLite sampler, custom binary reader, old training entry point and
GPT-2-specific launch/manifests have been removed from active source.
The old sampling process was explicitly stopped and verified at 08:42 UTC;
its partial data remains on B200, unused. No data, caches or checkpoints were
deleted. GPU burns were not touched.

Packing intentionally follows sparse embedding: no inserted EOS, batch-local
tail dropping and cached Arrow datasets. Keep preprocessing batch/worker counts
fixed across arms. Existing English eval is about 10M tokens, not the superseded
20M validation + 20M test plan. No separate final-test or new duplicate audit is
claimed. Preserve the common shifted-loss and exact-evaluation correctness fixes.

Local Qwen rewrite verification: 42 tests passed; all-six-arm BF16 CPU and
two-rank Gloo model smoke passed. Full two-rank Trainer/preprocessing/save/eval
smoke for D passed; its NLL matched standalone evaluation within 8.1e-8 on
exactly 378 synthetic scored targets. Single-process tests cover every arm,
cached multiprocess packing, exact resume, and standalone evaluation agreement.
These are local correctness tests, not evidence of B200 speed or model quality.

## Historical results and decisions (superseded Llama/GPT-2 implementation)

Local implementation verification (2026-09-07): initial 39 unit/integration tests passed,
including production meta-device parameter counts, all six tiny model arms,
causality, KV-cache/full-sequence equivalence, independent/tied save/load,
gradient checkpointing and exact interrupted-resume versus uninterrupted A128
weights. Tiny CPU BF16 smoke and two-rank CPU/Gloo smoke passed for all six arms.
The distributed smoke compared every parameter across ranks after different
rank-local batches. Reports: `temp/capacity_bf16_cpu.json` and
`temp/capacity_ddp_cpu.json`. These are correctness results, **not research
accuracy, speed, CUDA/NCCL or B200 memory results**.

An initialization bug was caught before handoff: nested HF pretrained-model
initialization initially bypassed the outer custom policy. Both body and causal
LM now share the same policy, with an explicit regression test. Real CulturaX
selection, tokenizer revision, near-duplicate audit, hardware smoke, and final
optimization settings remain prerequisites to research training.

The subsequent direct-NLL audit caught an HF 5.9 Trainer normalization mismatch:
its denominator included position zero, which is not a causal prediction target.
The training override now counts shifted targets. Validation now gathers
per-example NLL sums/counts to avoid the same issue and distributed final-batch
duplication bias. The earlier tiny `temp/capacity_ddp_trainer/run_D` result is
pre-fix diagnostic output, not a valid experiment result; use the `_correct`
rerun and final regression log. No real-data research training was affected.

Final verification: **40 tests passed** (`temp/capacity_all_tests.log`). The
corrected two-rank D trainer reported NLL 4.4567799892 on exactly 707 synthetic
validation targets; standalone evaluation reported 4.4567802118 on the same
targets (difference 2.23e-7, float32 reduction tolerance). Evidence:
`temp/capacity_ddp_trainer/run_D_correct/result.json` and
`temp/capacity_ddp_trainer/nll_audit.json`. These tiny synthetic numbers are only
an evaluator-consistency check, not model-quality results.

| Date (UTC) | Run / commit | Evidence location | Result | Decision |
|---|---|---|---|---|

## Operational lessons

### Historical SWT environment and English preparation, 2026-09-07

Created `swt` by offline-cloning each machine's own `sparse_emb`, preserving
installed package versions and leaving the source environments untouched.
Dev prefix: `/home/users/thien/miniconda3/envs/swt`; B200 prefix:
`/mnt/local/conda-py311/envs/swt`. B200 retains torch 2.14.0; dev retains
2.7.1+cu118. Both use Transformers 5.9.0 and Accelerate 1.13.0.
B200 Stagewise hashes matched and 40 CPU tests passed; the subsequent dev
preparer review passed 42 tests plus a real pinned GPT-2 packing smoke with
documents longer than 2048 tokens. No destination-GPU test was performed.

Reusing 50 English CulturaX raw shards (122,167,447,875 bytes), with hashes
matched against the pinned public release's Hub LFS metadata. Only the five
pinned GPT-2 tokenizer/config files were downloaded, not model weights.
The user requested **10B**, replacing the earlier 5B available-data target.
Registered offline command, seed/fractions, exact revisions, fresh output and
validation conditions are in `CAPACITY_EXPERIMENTS.md` and
the now-removed preparation script (recoverable in Git). This did not authorize training or extend
the screening optimizer schedule. All existing GPU work is left untouched.
Preparation completion and the required near-duplicate audit remain separate
checks; neither is claimed by the setup/smoke results.

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
