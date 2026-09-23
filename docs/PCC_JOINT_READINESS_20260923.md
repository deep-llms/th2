# Joint-training readiness review — 2026-09-23

**Ready for the bounded local joint-v1 pilot.** No six-run scientific training
job has been launched. B200, `prepare_data.py`, and `commands.sh` were untouched.
The review found that full-backbone joint training was still only a plan;
the existing implementation trained adapters with frozen/detached backbone
states. A separate joint-training implementation is now present and tested.

## Implementation and review

- `pcc/joint_model.py`: trainable fp32 backbone with bf16 autocast, all 28
  pretrained layers retained, differentiable shallow/deep sources, strict-past
  branch, and activation checkpointing without capture hooks.
- `CorrectionAdapter.forward` retains detached inputs by default for frozen PCC;
  the joint path explicitly opts into input gradients. Existing frozen behavior
  is covered by the regression suite.
- `pcc/joint_training.py`: global target-normalized loss, matched exact update
  counts, backbone/branch LR groups, fixed-set validation, atomic full-state
  checkpoints, explicit resume, and checkpoint-content auditing.
- `pcc/joint.py` / `joint_config.py`: separate CLI and fixed experiment settings,
  shared data preparation, six-run sequential runner manifest, paired final
  comparisons, CSV/PNG curves, and JSON/Markdown results. Scientific runs reject
  capacity-check checkpoints as resume inputs.
- Resume checks code/config/data identity, optimizer recipe/steps/moments,
  training cursor/history, finite weights, and tied embedding consistency.
  Final reports inspect checkpoint contents rather than trusting filenames.
- Added `matplotlib==3.10.9` to `train_env` and the dependency reference after the
  report test exposed its absence. `pip check` passed; `sparse_emb` is unchanged.

## Verification evidence

Full offline CPU regression: **109 tests passed in 94.189 seconds**, no failures
or skips. Evidence: `temp/pcc-joint-regression-20260923-a02.log`. The 15 joint
tests cover native no-op equivalence in fp32/bf16, gradient flow into deep source
states, actual early/late backbone and branch updates, checkpointed versus
direct gradients, accumulation, causal/segment masking, native token loss,
exact resume, corrupt-state rejection, seed-pool equality, runner manifest,
and paired report/plot generation. Standalone joint-test evidence:
`temp/pcc-joint-tests-20260923-a05.log`.

An initial accumulation assertion detected tiny fp32/AdamW differences near zero
gradients across batch shapes. The normalization test now uses SGD with the
original tight parameter tolerance; AdamW updates/moments and resume are tested
separately. All scientific arms use the same microbatch. The missing plotting
dependency failure and earlier test logs were retained, not relabeled as passing.

All three full-size checks used actual pinned pretrained weights and local
CulturaX contexts on dev host `transformer1`, physical GPU 0 (A100-PCIE-40GB),
microbatch 1, 2048-token contexts, and exactly 32,768 input tokens per update.
Each arm completed two diagnostic optimizer updates, initial native-equivalence
checks, fixed-set validation, actual parameter-change checks, and a full
checkpoint roundtrip. Their checkpoints are diagnostic artifacts only.

| Arm | Peak allocated GiB | Update 1 seconds | Update 2 seconds | Result |
|---|---:|---:|---:|---|
| Base | 10.810 | 6.682 | 6.546 | Pass |
| Shallow | 10.932 | 6.495 | 6.615 | Pass |
| Deep | 11.338 | 9.571 | 9.568 | Pass |

A further real-GPU Deep test compared three uninterrupted updates with resuming
the saved update-2 checkpoint and executing update 3. **The next-step NLL and
all model weights were bit-identical.** It executed four additional diagnostic
updates; evidence: `temp/pcc-joint-ready-20260923-a01/gpu-resume.json` and its log.
All checks together executed ten diagnostic updates / 327,680 input tokens,
plus validation. These are correctness/capacity checks, not method-benefit tests.

The measured training steps imply about **19.4 hours of optimizer-update time**
for six 1,536-update runs sequentially on this A100. Allow additional time for
validation, checkpoint writes/audits, input loading, and contention; roughly
20–24 hours is a planning estimate, not a runtime guarantee. No B200 is needed
for this bounded pilot. Two updates do not establish long-run convergence.

## Inputs and executable handoff

Local root: `temp/pcc-joint-ready-20260923-a01/`.

- `config.json`: actual local model/train/validation paths, microbatch 1.
- `source.json`: original pinned dataset source and document ranges.
- `inputs/complete.json`: fixed configuration, exact counts, and fingerprints.
- `jobs.json`: six sequential training jobs, preceded by CPU input preparation
  and followed by a CPU report. Repreparation uses the same fixed raw inputs;
  the ready input cache can also serve direct single-arm CLI calls.
- `capacity-{Base,Shallow,Deep}/capacity.json`: full-size check results.
- `gpu-resume.json`: full-size replay equivalence.

Training source rows: [0,20000) plus [30000,110000) of the already downloaded
English shard; validation remains [20000,30000). Training uses 24,576 contexts,
50,331,648 input tokens and 50,307,072 targets. Validation uses 977 contexts,
2,000,000 input tokens and 1,999,023 targets. No source rows overlap; no short
training stream is recycled. The validation fingerprint is unchanged from the
earlier local screen, so this remains exploratory validation.

When the local scientific launch is requested, use a fresh output directory:

```bash
/home/users/thien/miniconda3/envs/train_env/bin/python -u run_experiments.py \
  --config temp/pcc-joint-ready-20260923-a01/jobs.json \
  --project-dir /disk/thuat/deep2shallow \
  --run-dir temp/pcc-joint-training-20260923-a01
```

This command has **not** been executed. The runner rechecks GPU 0 before launch;
readiness does not reserve the device. Details, explicit resume behavior, and
output interpretation are in [the operating guide](PCC_JOINT_TRAINING.md).
