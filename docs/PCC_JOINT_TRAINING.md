# Joint full-model training

`python -m pcc.joint` implements the separately versioned joint-v1 experiment
in [the agreed plan](PCC_JOINT_TRAINING_PLAN_20260923.md). It trains all backbone
parameters from the pinned Qwen3-0.6B-Base snapshot. The existing `pcc pipeline`
continues to implement the frozen-backbone experiment; it is not this entry point.

The three arms are Base, Shallow and Deep. Each starts from a fresh copy of the
same pretrained checkpoint. Shallow/Deep share branch initialization and use
post-block pair (4,20). Deep's clean source and corrected tail both receive LM
gradients through the current model. No teacher alignment loss is used here.

## Data and fixed budget

Copy `pcc.joint.example.json` and supply existing model/train/validation paths.
The shared direct loader follows the existing packing/tokenization policy. No
separate user-facing raw-text export is needed. The preparation job validates
inputs and writes internal shared NPZ/cache files beneath the run directory.

Each arm receives 1,536 updates x 32,768 input tokens = 50,331,648 tokens.
Context length is 2048; microbatch must divide 16 and is fixed across arms.
The two seed pairs are (2901,20260922) and (3901,20260923). The second data order
permutes the exact same selected training-context pool. Validation is fixed
across all arms and seeds. Too-short data fails before any optimizer update.

Each arm records validation at step 0 and every 128 steps on 262,144 input
tokens; final evaluation uses 2M input tokens. Final-step checkpoints determine
the comparison. There is no automatic extension or arm-specific early stopping.

## Sequential launch interface

Generate and inspect a manifest; generation itself does not train:

```bash
conda run --no-capture-output -n train_env python -m pcc.joint manifest \
  --config temp/joint-config.json --physical-gpu 0 --output temp/joint-jobs.json
conda run --no-capture-output -n train_env python run_experiments.py \
  --config temp/joint-jobs.json --list
```

When launching an authorized local experiment, use a fresh run directory:

```bash
conda run --no-capture-output -n train_env python -u run_experiments.py \
  --config temp/joint-jobs.json --project-dir "$PWD" \
  --run-dir temp/joint-training-001
```

The manifest runs input preparation, six GPU jobs sequentially on the explicit
physical GPU, then a CPU report job. It stops on the first software failure.
The runner verifies completion artifacts and checks GPU availability before
and after each GPU job. It never launches on B200 or selects a remote host.

## Bounded capacity checks and resume

After CPU preparation, run `pcc.joint capacity` with the same config, data-dir,
arm and physical-gpu arguments as training. This performs exactly two updates
at the real global batch/context size, validates initial native equivalence,
checks early/late/branch parameter changes, and roundtrips a full checkpoint.
It writes `capacity.json` and `stopped.json`, never training `complete.json`.
Capacity checkpoints carry a distinct purpose and cannot resume a scientific
run. Their weights are not used as training initialization.

Scientific checkpoints include full model/branch weights, AdamW state, full
schedule configuration, exact completed update and next-context cursor, RNG
states, input fingerprints, source-weight hash, software versions, code hashes,
and training/validation history. `latest.pt` is written atomically every 128
updates. On completion, `final.pt` is a hard link to the final latest checkpoint;
it does not require a second full copy. Final weights/moments and budgets are
audited before the per-arm completion marker is published.

Explicit resume takes a **fresh output directory**:

```bash
conda run --no-capture-output -n train_env python -u -m pcc.joint train \
  --config temp/joint-config.json --data-dir temp/joint-training-001/inputs \
  --arm Deep --seed-index 0 --physical-gpu 0 \
  --resume temp/joint-training-001/seed-0-Deep/latest.pt \
  --output temp/joint-resumed-Deep-001
```

Resume refuses changed code/config/data/model identity, corrupt moments, wrong
optimizer steps, and inconsistent tied embeddings. It resumes the original
schedule at the saved cursor; uncheckpointed later work is repeated. Historical
logs are retained in the new output. There is no automatic retry or runner-wide
resume: after a failure, a new explicit continuation manifest must reference
the completed/resumed arm directories consistently before producing a report.

## Outputs and interpretation

Each arm saves `identity.json`, `invocation.json`, `train.jsonl`,
`validation.jsonl`, checkpoint(s), `eval.npz`, and a completion/failure marker.
The final report verifies all six runs, exact budgets, input/code/weight
identities, raw evaluation counts, and checkpoint contents. It saves JSON and
Markdown results, training/validation CSVs, validation-loss and contrast plots,
and timing/memory records. Context bootstrap intervals and both seeds are
reported separately.

A positive next-stage recommendation requires Deep to beat both trained
controls in both seeds with upper paired 95% CIs below zero and at least
.0005 nats/token improvement over Shallow in each seed. It triggers no further
training, new layer search, or remote submission. The reused local validation
split makes this exploratory evidence, not an independent confirmatory test.

## Bounded document selection for the B200 pilot

Optional joint config `train_documents` selects a fixed uniform document pool
without replacement from the completed training split before tokenization.
The deployment uses 100000, seed 20260922, sorted selected indices, followed by
the existing context packing/shuffle. Indices and hash are saved with the input
cache; all arms reuse the same pool. Evaluation is unaffected. Omitting this
field preserves full-split preprocessing. A short pool fails; no repetition or
automatic expansion occurs. This prevents preprocessing the entire 36.6M-document
corpus for a 50M-token run. See the pre-launch amendment in the experiment plan.
