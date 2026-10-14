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

Results: pending; do not interpret implementation tests as scientific evidence.
