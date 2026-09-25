# Four-A100 restart after loss of the B200 worker

The user authorized this restart on 2026-09-25. The B200 trained checkpoints
and prepared inputs were not downloaded and are unavailable. Compact results
remain valid records of the earlier experiment, but cannot initialize students.
This is a separately versioned local experiment, not a continuation or exact
reproduction of the B200 data split. No remote runner submission is needed.

## Fixed design

- Full 28-layer pretrained Qwen3-0.6B-Base, revision ddc928429ed09d9ad603fd762053d0434c15e865.
- `joint-local-v3`: first train one Deep teacher, seed 2901 / data seed 20260922,
  with the same joint optimizer and schedule as joint-v2: 6144 updates,
  32768 input tokens/update = 201326592 input tokens. s=4,d=20.
- Every experiment uses physical GPUs 0,1,2,3 sequentially. Microbatch 1,
  context 2048, four accumulation steps per rank, 16 contexts globally.
  Frozen students keep this batch, budget, and 307-step warmup.
- Local source is the verified CulturaX English parquet en_part_00015, revision
  b19d850278693d37113c197857cc6328fa5c6881; 1106987 rows. Train rows
  [0,20000) plus [30000,610000): 600000 documents, no replacement. Validation
  rows [1096987,1106987): 10000 documents not used by the earlier local runs.
  The previous local validation rows [20000,30000) are excluded from training.
  Row disjointness does not establish corpus-wide semantic deduplication.
- Existing legacy 1000-document packing, pinned tokenizer, no special tokens,
  seed 20260922 context shuffle, fixed first 201326592 train input tokens and
  2000000 dev input tokens. Fail on shortfall; no recycling/resampling.
- Verify the teacher's final checkpoint by reloading and reproducing its full
  dev evaluation. Evaluate the same trained backbone with feedback off.
  Require benefit >=0.0005 nats/target and paired 95% CI entirely below zero.
- Only if the feedback gate passes: `distill-local-v2` LM-only shallow student
  and PCC student, byte-identical fresh initialization, frozen shared parent.
  Distill the teacher's block-4 correction with the existing normalized SmoothL1
  objective; the student sees shallow states only. Same 1M-token calibration,
  lambda=1, beta=1, adapter LR3e-4, checkpoints/monitor every256 updates.
- Require PCC to beat LM and feedback-off with paired 2000-resample 95% CIs,
  and recover >=25% of the feedback benefit. No cross-split comparison against
  the old B200 Base model. We are not retraining Base/Shallow controls here.
- Only a passing first student pair permits the second teacher/student seed
  (3901 / 20260923), using a permutation of the same fixed training pool. Report
  both seeds; a negative teacher or student gate stops successfully as a
  scientific result. Software failures stop with an error, no automatic retry.
- At most six scientific runs (two Deep teachers, two LM and two PCC students).
  First seed costs three runs if its feedback gate passes. No test set unlock,
  automatic target-semantics experiment, or adaptive budget extension.
  This remains exploratory, including the conditional second-seed decision.

## Execution and durability

Runtime: /home/users/thien/miniconda3/envs/train_env/bin/python.
Root: /disk/thuat/deep2shallow/temp/local-restart-20260925-a01.
Separate filesystem backup: /home/users/thien/deep2shallow-backups/local-restart-20260925-a01.
Home is a network mount (`cranium:/p/research/legendaryhome/users`); this protects
against loss of the dev machine's local disk, not every possible storage failure.
No claim is made about the storage service's own snapshot/retention policy.

The source is copied to root/source after passing CPU tests. The CPU readiness
receipt records the exact pcc source hashes. scripts/local_restart.py checks it,
backs up source/config/data/model assets with verified SHA256, then performs a
real four-rank Deep capacity and resume test before scientific training. LM/PCC
capacity/resume checks precede their first scientific runs. The controller runs
in tmux, logs each foreground stage, and saves pipeline.json status. Teacher
training uses run_experiments.py with a one-job manifest; the controller gates
subsequent stages on validated scientific receipts.

All trainer checkpoints are atomic replacements. A backup thread opens the
current checkpoint inode, copies and fsyncs it, independently hashes the copied
bytes, then atomically publishes the network copy and manifest. It polls every
15 seconds; checkpoints are written every256 updates. A failed backup prevents
the next stage. A completed teacher's full final.pt and receipts are backed up
before any student audit/training. Inputs, source, tokenizer and pretrained model
are also copied. Backups retain original absolute paths and hashes; restoration
must preserve those paths for strict resume or explicitly audit any relocation.

Progress: pipeline.json, teacher-seed-0/runs/seed-0-Deep/train.jsonl and
validation.jsonl, and the network backup manifest.json. Scientific stops create
root/complete.json with a stop decision. Failures leave pipeline.json failed.
The local controller launches no burn and never stops unrelated GPU processes.

## Validation observations

The initial expanded CPU suite ran 129 tests: 128 passed; the new four-rank Gloo
resume test failed its bitwise post-update comparison (reported max difference
9.31e-10). Reloaded weights and RNG are now checked bitwise, while subsequent
four-rank updates use atol 1e-8 / rtol 1e-6 to allow floating-point collective-order
variation when DDP rebuilds buckets. The existing exact two-rank check remains.
No training formula or scientific gate changed. Initial failure log is retained
as regression.log. Local parent-checkpoint, backup corruption, and negative/
positive sequential-gate tests passed 4/4; updated report tests passed 2/2.
A final full regression and real four-A100 capacity/resume check are required
before scientific startup; no bitwise cross-topology reproducibility is claimed.

Final CPU regression passed **131 tests in 165.407 seconds**, offline with GPUs
hidden (`regression-v2.log`). The final controller-only gate/backup test also
passed (17.528 seconds, `queue-tests-final.log`). The expanded distributed tests
passed 3/3 in 44.976 seconds. Scientific startup remains gated by the actual
four-GPU capacity/resume receipt. No B200 checkpoint is reused.


Live launch: source commit 7ec9b4f, tmux pcc-local-restart-20260925-a01,
controller PID 771522 (verify PID identity before any action). Initial backup
completed with hashes matching all pinned model assets. Four-GPU capacity
passed with identical replica hash 059d6988..., 3.216/2.880 seconds per update,
and resumed update 3 completed. Peak allocated during capacity/resume was
21729107968 bytes. The controller advanced to teacher-seed-0. These are
readiness results, not scientific evidence for the method.

Fixed input fingerprints: train
0f79e3ad69b8af9d4aec7636f05a0492b6e09b2ecdcbef8bf369cc109dec9bb1;
dev dbcc5234be00e7b82682112922a496f00ffa5359c9b75106d064568e67685e9e.
98304 training contexts / 201326592 input tokens; 977 dev contexts / 2000000 inputs.


At 2026-09-25 04:40:49 UTC, seed-0 Deep reached update 14/6144 with finite
loss/gradient norms and 2.9–3.1s/update. Workers 771852–771855 occupied GPUs 0–3
at about 15.3 GiB each. invocation.json confirms resume=null and stop_after=null.
Initial fixed monitor NLL was 3.0760444572. No final scientific result yet.
