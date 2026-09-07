# Project notes

Status: v0.5 English capacity-allocation pilot implemented locally; no research
training results. The user explicitly assigned the existing th2 B200 to this
project on 2026-09-07; no Stagewise workload has been launched.

## Purpose and success criteria

Implement and test §12 of `Capacity_Allocation_Research_Project_v0.5_Final_Reviewed.md`:
random-initialized Llama-family B0/A128 first, A256/A512 controls, then C/D T1
stagewise width schedules. See `CAPACITY_EXPERIMENTS.md` for exact code and usage.
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

## Results and decisions

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

### SWT environment and English preparation, 2026-09-07

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
`scripts/prepare_english_b200.sh`. This does not authorize training or extend
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
