# Document-isolated attention: 200-update comparison (2026-10-04)

Both training jobs completed 200 updates and saved their final weights. The
predeclared trajectory comparison **failed**. No threshold was relaxed and no
production attention implementation was changed.

## Matched setup

Full randomly initialized Qwen3-0.6B arm A, seed 42, eight B200s per run,
BF16, microbatch 16, accumulation 4, sequence length 2048, 1,048,576 input
tokens per update. Decoder/LM activation checkpointing off. Original
28,600-update schedule and 1,430-update warmup retained. Both tests therefore
remain in warmup. This is pretraining from random initialization, not a
pretrained-weight finetuning test.

The same 8,192 packed training rows (16.78M input tokens) repeat for 12.5 epochs.
A separate 512-row validation pool provides 1,046,868 prediction targets.
Appended EOS and source document IDs define boundaries; both implementations
block attention across documents. Same seed, optimizer, learning rates, and
entire per-rank token/document-ID streams verified. All 16 rank receipts show
200 updates and 12,800 observed rows. All logged losses and gradient norms are
finite. All 24 exported source artifacts passed size/SHA256 verification.

## Measurements

| Measurement | Dense isolated SDPA | Isolated FA4 |
|---|---:|---:|
| Final training loss | 7.1168804 | 7.1171484 |
| Held-out loss, common dense backend | 7.1107591 | 7.1074235 |
| Median seconds per update | 2.42939 | 2.07476 |
| Peak allocated GPU memory, GiB | 110.62 | 103.62 |

Update times include forward/backward, loss, DDP, optimizer and data delivery.
They exclude startup, the first five updates, evaluation and saving. Each
interval uses the slowest rank. FA4 uses 14.6% less update time in this pair;
this is one run per implementation, not a statistical performance claim.

| Predeclared comparison | Observed | Limit | Outcome |
|---|---:|---:|---|
| Maximum absolute training-loss gap | 0.01066065 | <0.01 | Failed |
| Maximum relative gradient-norm gap | 48.098% | <3% | Failed |
| Held-out loss gap, common backend | 0.00333556 | <0.01 | Passed |
| Same FA4-trained weights, dense/native evaluation gap | 0.00000258 | <0.01 | Passed |

Through update 100 the maximum loss gap was 0.00001621 and the maximum relative
gradient-norm gap was 0.00628%. The gradient-norm threshold was first exceeded at
137 and was exceeded at 49 updates overall. At update 196 the dense/FA4 norms
were 1.76583/0.91649 and losses were 7.18331/7.17265: the largest gap in both
metrics. Only this update exceeded the training-loss threshold. At update 200
the norms still differed by 42.44%, so this is not a single isolated outlier.

## Interpretation and next check

Both runs learned and ended with close held-out losses. They did not maintain
the strict trajectory agreement seen in the 30-update test. These gradient
norms belong to **different trained weight sets**; their difference is not a
measurement of FA4's gradient error at identical weights. Accumulated numerical
drift is a plausible explanation, but this experiment cannot establish the
cause. Nor does the small held-out advantage establish that FA4 improves learning.

The initial same-weight full-gradient correctness check passed again. At the
end, the same FA4-trained weights evaluated through dense SDPA versus FA4 gave
losses 7.10742352 versus 7.10742610. This supports forward agreement on this
held-out pool, but does not verify backward agreement at the trained weights.

The next focused check is full-parameter gradients using the same saved model,
batch and masks with both backends, ideally including a higher-precision
reference. A repeated dense-backend training control would help establish how
much trajectory variation occurs without changing the attention backend.
Neither follow-up was launched. Do not promote FA4 to production on a claim
that this strict 200-update comparison passed. Auxiliary arms also need their
own integration checks.

## Operations and evidence

Job: th2-tjx3-document-stability-200-20261004-a01, submitted e06d5f9.
Remote output: /mnt/local/_outputs/deep-llms_th2/document-stability-200-20261004-a01.
Both training jobs exited successfully. The summarizer wrote comparison.json
with passed=false, then raised its gate assertion; summary.json was not written.
The supervisor correctly records passed=false. Original receipts are preserved.

Eight GPU burn workers 17301–17308 were automatically restored, with collective
progress verified at 2026-10-04 13:30:02 UTC (21:30:02 Singapore). The read-only
collector also verified matching live workers and a released GPU guard. The
separate attention_bench environment was used; original train_env and driver
were unchanged. Accelerate config copying/env validation and GPU reclamation
were verified before the training launch.

Local evidence: artifacts/document-stability-200-20261004-a01/. Includes raw
comparison.json, correctness.json, both trainer states, 16 rank receipts,
result.json, and the source manifest. Additional analysis.json and
training_comparison.png/.svg were generated locally without changing raw files.
Final model weights remain on B200; they were not pulled in the small archive.
Archive SHA256: 638bf5cec754e40699996d5b4c8f8261337958c2d77191ca9c9d521abe9eacba.
Completion log SHA256: 90ba1d61684da05fc82255e45d2f331b0e7533377e85e6dbd7b0250cb3d3aa87.
