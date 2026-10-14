# Does the current consumer teacher help next-token prediction?

Decision: whether to keep investigating the current teacher-response objective,
or change that objective before spending compute on more training variants.
No new training, downstream sweep, environment installation or checkpoint edits.

Frozen scope: existing 12-layer, seed-42, step-2500 checkpoints of P7-simple,
P7-simple-lm, P7-simple-deep and P7-simple-consumer (consumer plus cosine).
Exact paths/hashes: `resources/p7_teacher_probe_20261015.json`.

For each validation batch, record the unmodified model's shallow inputs,
attention outputs and standardized/clipped four-MLP-sum targets. Construct the
teacher's last-four-head attention response using the existing native Q/K/V,
proxy projections, document isolation and FA4 path, exactly as the consumer
loss does. At **one site at a time**, replace those head outputs by

`student + alpha * (teacher - student)`

before the output projection. Earlier computation stays identical; downstream
computation runs again normally. The other sites continue using their learned
predictors. Sites: blocks 2, 4, 6 and 8. Alpha: +0.1, -0.1, 1.0. The opposite
direction is an equal-magnitude control, not another training arm.

This is a **local frozen-teacher counterfactual**, not recurrent real-target
inference or an oracle bound. The teacher is held fixed from the unmodified
trajectory; it is not claimed to be the modified trajectory's own deep target.
We intervene at the attention response because that is precisely what the
consumer objective asks the predictor to match. We do not change the deep
adapter input in P7-simple-deep.

Use 512 deterministic, evenly spread packed rows from the original validation
dataset: 1,048,576 input tokens per checkpoint, same EOS boundaries/reset
positions and scored-target rules as training. Verify its original fingerprint.
Split alternating selected rows into two disjoint 256-row halves; four models
times two halves run concurrently on eight GPUs. Each worker loads once per
stage. A two-row-per-worker CUDA smoke must pass before the full probe.

Checks: independent eager teacher and downstream reference locally, original
forward and alpha-zero exact identity, native heads unchanged, teacher response
equal to the training consumer objective, future/other-document invariance,
state/checkpoint hashes unchanged, exact row coverage. No optimizer exists.

Interpretation, declared before results:

- +0.1 improves loss, -0.1 hurts, consistent in both halves: the teacher is a
  useful direction at that site; better learning or utilization remains plausible.
- Small movement helps but full replacement hurts: compatibility/strength is
  a limitation; full-target replacement alone is a misleading test.
- Movement toward the teacher fails while movement away helps: current
  response matching is locally misaligned with LM loss; avoid a blind weight
  sweep and reconsider the teacher/objective.
- Mixed or tiny effects: inconclusive. Do not generalize a local result into
  impossibility of proxy methods or predict unseen downstream gains.

Report paired loss deltas in nats per scored token, both halves, and descriptive
paired packed-row bootstrap intervals. These intervals are exploratory, not
independent-seed evidence, document-cluster intervals or multiplicity-adjusted
significance. No architecture is selected solely from the best of 16 sites.

Implementation: `eval/p7_teacher.py`, `scripts/probe_p7_teacher.py`.
Launch uses the established verified-burn supervisor, Accelerate config copy
and verification, exact known-worker reclamation, all-GPU free check, then
automatic communicating-burn restoration on success or failure.

## Results — 15 October 2026

**Decision: do not launch a larger auxiliary-weight/target-matching sweep on
this evidence.** Matching the current teacher response more strongly has not
shown a useful LM-loss improvement at these trained weights. This result does
not invalidate proxy architectures or exclude improvement after retraining.

All local checks and eight-worker CUDA controls passed. Both identity replays and
the comparison with the actual consumer objective had exactly zero error in all
16 smoke/full worker reports; all model/checkpoint state checks passed. The full probe evaluated
512 rows / 1,046,840 scored tokens per checkpoint. Smoke took 36.31 seconds;
full evaluation took 57.89 seconds on all eight GPUs, including worker/model
startup. Both stages completed at 07:12:21 SGT. No training was performed.

Full replacement worsened loss at **all 16 model/site combinations**, in both
halves; every descriptive paired-row interval was above zero. At 10% strength,
none of the negative deltas had an interval wholly below zero. P7-simple-lm
was worse at sites 2/4/6 even at 10%; P7-simple-deep was worse at site 4.
The other small changes were inconclusive. No reverse-direction condition
showed an established improvement either: the data do not support saying that
moving away from the teacher is generally beneficial.

| Checkpoint | 10% movement: range across sites | Full replacement: range across sites |
|---|---:|---:|
| P7-simple | -0.0000283 to +0.0000358 | +0.001252 to +0.007874 |
| P7-simple-lm | -0.0000181 to +0.0008526 | +0.003201 to +0.137619 |
| P7-simple-deep | +0.0000083 to +0.0001142 | +0.000976 to +0.007389 |
| Consumer + cosine | -0.0000104 to +0.0000254 | +0.000271 to +0.003003 |

Units: change in nats per scored token; **negative is better**. Each entry is
relative to its own unmodified checkpoint. These ranges are four separate
single-site interventions, not a simultaneous intervention across all sites.

The practical lesson is narrower than “deep targets contain noise”: the current
teacher response is not demonstrably better for the trained consumer. The model
has learned to use its predicted memory, and substituting the deep-target response
is not a reliable improvement. Consumer+cosine is more tolerant of replacement,
but tolerance is not evidence of a better teacher. The LM-trained predictor's
large full-replacement penalties further caution against judging its usefulness
by target similarity alone.

A very small move below 10%, a different teacher, a joint intervention, or a
model adapted by retraining could behave differently. This cheap test does not
resolve those possibilities. They do not justify another broad training grid
without a new hypothesis. Keep existing checkpoints as references; if exploring
further, change how a useful teacher is defined or selected, rather than assuming
that stronger imitation of this one is the missing ingredient.

## Per-site paired deltas

The 95% intervals below are descriptive packed-row bootstrap intervals, with the
limitations stated above. The two disjoint row halves agree on harm for every
full replacement. All raw per-row records remain under the remote run root.

| Model | Site | Toward 10% | Opposite 10% | Full teacher | 95% interval for toward 10% |
|---|---:|---:|---:|---:|---|
| P7-simple | 2 | +0.0000358 | +0.0000772 | +0.0078738 | [-0.0000097, +0.0000829] |
| P7-simple | 4 | +0.0000021 | +0.0000638 | +0.0046707 | [-0.0000420, +0.0000470] |
| P7-simple | 6 | -0.0000023 | +0.0000132 | +0.0016980 | [-0.0000363, +0.0000317] |
| P7-simple | 8 | -0.0000283 | +0.0000251 | +0.0012517 | [-0.0000622, +0.0000046] |
| P7-simple-lm | 2 | +0.0004648 | +0.0007052 | +0.1376191 | [+0.0003669, +0.0005699] |
| P7-simple-lm | 4 | +0.0008526 | +0.0003501 | +0.1010295 | [+0.0007512, +0.0009509] |
| P7-simple-lm | 6 | +0.0004666 | +0.0000913 | +0.0902068 | [+0.0003834, +0.0005532] |
| P7-simple-lm | 8 | -0.0000181 | +0.0000848 | +0.0032005 | [-0.0000568, +0.0000197] |
| P7-simple-deep | 2 | +0.0000083 | +0.0001558 | +0.0071666 | [-0.0000427, +0.0000598] |
| P7-simple-deep | 4 | +0.0001142 | +0.0000323 | +0.0073889 | [+0.0000646, +0.0001635] |
| P7-simple-deep | 6 | +0.0000160 | +0.0000352 | +0.0018228 | [-0.0000209, +0.0000526] |
| P7-simple-deep | 8 | +0.0000168 | +0.0000286 | +0.0009759 | [-0.0000171, +0.0000502] |
| Consumer + cosine | 2 | -0.0000058 | +0.0000502 | +0.0030027 | [-0.0000479, +0.0000359] |
| Consumer + cosine | 4 | -0.0000104 | +0.0000616 | +0.0007967 | [-0.0000464, +0.0000287] |
| Consumer + cosine | 6 | +0.0000254 | +0.0000026 | +0.0003711 | [-0.0000040, +0.0000566] |
| Consumer + cosine | 8 | -0.0000099 | +0.0000216 | +0.0002711 | [-0.0000390, +0.0000190] |

Evidence:

- Launch 87f146a; read-only receipt 45a1567 at 07:13:09 SGT.
- Root:`/mnt/local/_outputs/deep-llms_th2/q359-p7-teacher-probe-20261015-a01`.
- `supervised/run/full/summary.json` SHA256:
  `0acb49fc27dacf0daab550e6c242cedf6e29fb56fcdf1c8fa0363201fa86e269`.
- `supervised/run/smoke/summary.json` SHA256:
  `f4fe8d7b708aa3ed2cd825813993ce1c23c5527f280af59b43599ab9af8910ab`.
- Pulled summaries reconstructed byte-for-byte and verified against the runner's
  artifact hashes. Local:`temp/results/p7-teacher-full-20261015.json`.
- Receipt:`temp/remote_logs/p7-teacher-health-20261015-a01.log`, SHA256
  `41831a7f264b6a56a1a0d661e0d9ff1ca1f2e4652bcb1e4a930c462d7127f190`.
- Original validation fingerprint:`08432871cf987d61`. Accelerator config copied
  and verified. Known burn workers 816616–816623 reclaimed; all eight GPUs
  verified free before smoke. Supervisor restored and verified communicating burns
  at 07:13:32 SGT. Fresh ae122b5 receipt at 07:16:24 SGT confirms all eight live workers,
  matching ownership/approved parents, advancing collectives, log age 2.64 seconds
  and released GPU guard. No follow-up work queued.
- Completion receipt:`temp/remote_logs/p7-teacher-completion-20261015-a01.log`,
  SHA256 `c96a1bd0f8bab4aa54daaae48562c5876af754c5336125cc58128e09fd12b616`.
- The exact verified full summary is retained in
  [results/p7_teacher_probe_20261015.json](results/p7_teacher_probe_20261015.json).
