# FA4 save/resume verification on eight B200 GPUs

**Passed.** Both fresh-process checkpoint continuations exactly matched the
corresponding part of the uninterrupted run on all eight ranks. No model,
optimizer, attention-kernel or training-default changes were needed.

## Test

Used the real `train.py` entry point, HF Trainer/Accelerate, BF16 and the existing
FA4 4.0.0b33 environment on q359. The ordinary data/training recipe checks and
Trainer's checkpoint loading/data skipping remained active. Launch: `82f6428`.

The inexpensive synthetic fixture retains 12 layers, eight routed steps,
head dimension 128 and GQA. Width is 256, MLP width 768, vocabulary 512, with
4 query / 2 KV heads. Sequence length 128, microbatch 1, accumulation 2, eight
ranks. Data are packed through the normal document-stream code with EOS,
document isolation and reset positions. No pretrained weights or real corpus.

The full cosine schedule remains 200 optimizer steps with 20 warmup steps.
The test-only router prior is zero and exploration stays active through the
second half, using maximum epsilon 0.8 and end fraction 1.0. The soft phase ends
after 30 steps. Production routing defaults are unchanged. Activation
checkpointing is off, matching the previously benchmarked setup. FA4's existing
default backward mode is used; no deterministic override was introduced.

| Restart checkpoint | Compared optimizer steps | Compared microbatches per rank | Maximum loss difference | Result |
|---|---|---:|---:|---|
| 100 | 101–200 | 200 | 0.0 | Exact match on all 8 ranks |
| 20 | 21–40, crossing soft → hard | 40 | 0.0 | Exact match on all 8 ranks |

For all 1,920 compared microbatches across the eight ranks:

- Input/label hashes and Python, NumPy, CPU and CUDA RNG-state hashes match.
- Every selected routing branch matches, including the secondary soft branch.
- Losses match exactly, exceeding the specified absolute tolerance of 1e-6.
- Per-update LR, curriculum progress, balancing bias and routing counters match.
- Restored and final model, AdamW and scheduler states have identical value
  hashes on every rank. Full-schedule and stopping steps are correct.

The downloaded rank-0 traces were also compared independently on the dev machine.
The probe first passed a local CPU end-to-end test; its comparator rejects
deliberately corrupted data, RNG, routes, loss and restored optimizer state.

## Operations and evidence

Accelerate's resource config was copied and verified as eight-process BF16.
The supervisor stopped only verified burn workers and verified all GPUs free
at 19:48:20 UTC. The test and automatic burn handoff completed successfully at
19:53:15 UTC. A separate read-only inspection at 19:56:35 UTC verified all eight
burn workers at 100% utilization, advancing collective cycles (220→240), and
the GPU guard released. commands.sh is returned to #0.

- [All-rank comparison results](results/bdot_fa4_resume_20261018.json).
- [Final runtime/checkpoint/burn receipt](results/bdot_fa4_resume_handoff_20261018.json).
- Probe: `tests/bdot/resume_gpu_probe.py`; comparator tests:
  `tests/bdot/test_resume_probe.py`; queue: `resources/bdot_b200_resume.json`.
- Remote artifacts:
  `/mnt/local/_outputs/deep-llms_th2/q359-dot-resume-20261018-a01/`.
- Local traces/logs: `/disk/thuat/DOT/temp/bdot_b200_resume_20261018/`.

This closes the pending GPU/FA4 resume check for this tested configuration.
It does not promise bitwise equivalence after changing backend, GPU count,
precision, batch size or library versions, or for all kernel shapes. The earlier
SDPA-versus-FA4 near-tie routing caveat remains. Data selection and the full route
study exporter are separate remaining pilot tasks. No research training was queued.
