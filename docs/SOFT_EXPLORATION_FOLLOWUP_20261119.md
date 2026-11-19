# Soft-route follow-up: scheduled exploration floor

Status: local model/trainer/fullstate/preparation/evaluation/report/export/queue implemented. Four CPU model/primitive tests PASS; fullstate boundary and end-to-end pipeline tests are running. No remote admission or launch. Finish soft-route a02 compact export/hash/local replay before remote admission. Preserve a01 failure, a02 outputs and all previous studies. User authority is continued autonomous testing after negative screens; original idea is `docs/warmup_DOT_idea_sketch.md`.

## Evidence and hypothesis

Verified a02 node results meet no registered quality/mechanism threshold. The dynamic soft model gains roughly0.0001nat over dense, and its hard projection loses. Training argmax counts are almost entirely canonical: S0 has4off-diagonal decisions among131072000, S1zero. Initial canonical probability was0.999, selected to preserve the pretrained dense function. This observation does not prove that insufficient exploration caused failure: alternative layer compositions may simply be worse. Adam normalization also means small raw softmax derivatives alone do not demonstrate optimizer starvation.

Test whether forcing the bodies to experience noncanonical mixtures during a bounded middle part of training can make learned composition useful. Do not infer quality from route diversity alone.

## Locked comparison

Reuse the exact Horizon D80k parents, seed-specific64000recorded rows, batch32, length512/prefix32,2000updates, optimizer and LR schedules of a02. Reuse its C/G/S endpoints only as matched controls; verify actual parent, input order, recipe and source hashes. This repeats training examples deliberately to isolate the changed intervention; it is not additional unique-data evidence. Keep separate output/session/backup and all existing checkpoints.

Four new fits: global G-floor and causal token S-floor, each seeds0/1. Both evaluate all four middle candidates at each of four depths, as a02. Modify only forward training probabilities:

`p_used = (1 - epsilon(step)) * p_router + epsilon(step) / 4`.

Use zero-based update index s. Fix epsilon(s)=0.4*min(s/500,1,(1500-s)/500), clipped to[0,0.4]. Thus epsilon0=0, it rises linearly to0.4at500, stays0.4through1000, falls to0at1500, and remains0 for the final500updates. Each candidate receives at least0.1forward mass at the plateau. This is a deterministic exploration/compatibility curriculum, not an RL estimator, not discrete order, and not a token-sampling distribution. The router continues learning through the differentiable mixture. No entropy reward, temperature sweep, extra dataset, teacher loss or changed LR is added in this screen.

Report both intended-router and used-mixture gate probabilities, entropy and argmax counts; uniform probability flooring preserves argmax for a fixed logit vector, so the intervention directly forces mixture exposure, not discrete path exploration. At final evaluation epsilon is exactly0 for every checkpoint, and soft/hard modes use the same endpoint weights. No final-label selection or gate schedule adjustment.

## Tests and admission

Reuse the established pipeline with project-local adaptations, keeping the old frozen implementation intact. Tests must cover exact epsilon values/boundaries, normalization and floor, causal independence, double-precision gradients, global versus dynamic modes, exact dense hard identity at initial epsilon0, and full-state continuation across both ramp boundaries. The schedule and update position must be serialized/locked; replaying a checkpoint cannot restart the curriculum. Tiny integration must exercise real Horizon parent schema, actual four-fit entrypoint, both-domain soft/hard summaries, independent reference, report/export tamper rejection and9same-node backup copies.

Node CPU checks must load actual parents and prove row arrays match a02. Fresh source/data/runtime/ownership checks precede GPU work. Preserve original production gates: initial hard identity, initial soft NLL perturbation≤0.01nat at epsilon0, exact same-shape future independence, BF16 numerical gradient relative error≤0.08, FP32 fullset independent parity≤2e-4nat, first40fullstate hash/resume and terminal weight/state hashes. Audit used-mixture causality at epsilon0.4also. No relaxation of a failed gate; any change requires explicit documented new protocol and tests.

## Criteria and interpretation

Primary discrete screen: S-floor-hard must beat G-floor-hard and matched C by at least0.005nat in each seed and each domain. Report paired document CIs and all S-floor versus original S, and G-floor versus original G comparisons. For a dynamic-order mechanism claim, demonstrate noncanonical, input-dependent hard routes; quality gains with a canonical projection are only a training-regularization result. Soft quality requires analogous S-floor-soft versus G-floor-soft and C thresholds and remains a continuous-mixture result only. No winner selection among seeds or domains.

Report training/eval wall time, actual18candidateblock forwards versus C6, route statistics by depth, projection gap, and conditional480target-token NLL. Retain reused-final/multiple-comparison/two-seed/unequal-compute flags. No untouched-confirmation or sparse/decode speed claim. Only compact arrays/reports/proofs leave node; weights/fullstate remain node-only. Require report/replay/backup/native/owned-guard-release verification, retain the existing watcher, then reassess the next experiment from evidence.
