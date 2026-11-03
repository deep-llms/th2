# A2 Depth — Parts A and B

Spec SHA-256: `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`.

Part A source report: {'path': '/mnt/local/_outputs/deep-llms_th2/6y3o-a2-depth-parta-20261101-a01/results/a2_depth_20261031/REPORT.md', 'sha256': '6ae54a45836d2aa670824ac01b598bcac30754b4ad774d59519f0869952b16b5'}.

| Dataset | Block | E | E / Δ(P300) | A_init | A_init / Δ(P300) | E label |
|---|---|---:|---:|---:|---:|---|
| dclm | W1 | 0.028632244673585383 | 0.21539858152781002 | 0.0712501955104603 | 0.5360107536623143 | {'value': 0.028632244673585383, 'clauses': {'INCOMPLETE': False, 'persists': True, 'reversed': False, 'erased': False}, 'label': 'persists'} |
| dclm | W9 | 0.03303367332921914 | 0.11427491360908006 | 0.07713408873095728 | 0.2668335197905046 | {'value': 0.03303367332921914, 'clauses': {'INCOMPLETE': False, 'persists': True, 'reversed': False, 'erased': False}, 'label': 'persists'} |
| dclm | S1 | 0.021096293347514283 | 0.10302884126700326 | 0.07272470370653528 | 0.35516864649842933 | persists |
| fwe | W1 | 0.0345100244148524 | 0.2812193407371629 | 0.07505336227155057 | 0.611603654763835 | {'value': 0.0345100244148524, 'clauses': {'INCOMPLETE': False, 'persists': True, 'reversed': False, 'erased': False}, 'label': 'persists'} |
| fwe | W9 | 0.039414471399715545 | 0.15368555952561244 | 0.08178711849348197 | 0.318905686700369 | {'value': 0.039414471399715545, 'clauses': {'INCOMPLETE': False, 'persists': True, 'reversed': False, 'erased': False}, 'label': 'persists'} |
| fwe | S1 | 0.026224221566386152 | 0.1339870997016109 | 0.07847837360805077 | 0.4009686099708777 | persists |

Predictions: `{"B1": true, "B2": false}`. Every clause: `{"dclm": {"B1": true, "B2": false}, "fwe": {"B1": true, "B2": false}}`.

Scientific coverage complete: True. Flags: 0.

DT4 and scientific flags remain recorded; they do not cancel Part C. Ledger: ledger.json (report-only).

Remaining: publish this A+B result, DT6, three 12.8B fits, downstream, independent replay and final native handoff.

## Part A: completed depth table and A1–A4

The Part A numerical results below are retained from its earlier report; its then-pending-stage footer is omitted here.


Spec SHA-256: `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`.

Coverage complete: **True**. Saved arrays: 156/156 (144 pruned + 12 parents).

FP32 parameters rounded to BF16 saved precision, native BF16 AMP/SDPA; final CulturaX set, unchanged A2 scorer.

Paired sequence bootstrap: 10,000 draws, seed 0. Relative intervals recompute the reference damage in every paired draw.

| Dataset | Set | L_ann / D(P300) | L_stab / D(S72) | R_short / D(S72) | R_long / D(S188) |
|---|---|---:|---:|---:|---:|
| dclm | W1 | -0.231621 | -0.146985 | 0.141551 | 0.028286 |
| dclm | W2 | 0.047056 | 0.065850 | 0.605570 | 0.577259 |
| dclm | W3 | 0.245461 | -0.100480 | 0.409317 | 0.951317 |
| dclm | W4 | 0.088436 | 0.090806 | 0.036337 | 0.034086 |
| dclm | W5 | -0.007412 | 0.010100 | 0.047619 | 0.029456 |
| dclm | W6 | 0.066162 | 0.038535 | 0.139724 | 0.170043 |
| dclm | W7 | 0.229747 | 0.161430 | 0.194582 | 0.264850 |
| dclm | W8 | 0.284440 | 0.190261 | 0.084583 | 0.170400 |
| dclm | W9 | 0.289559 | 0.127370 | 0.121081 | 0.282366 |
| dclm | S2 | -0.231621 | -0.146985 | 0.141551 | 0.028286 |
| dclm | S3 | 0.140521 | 0.001845 | 0.148428 | 0.307394 |
| dclm | S4 | 4.008777 | 0.203851 | 0.806788 | 6.517374 |
| fwe | W1 | -0.296443 | 1.184652 | 1.007967 | -0.353344 |
| fwe | W2 | -0.160492 | -0.340845 | 0.342944 | 0.710391 |
| fwe | W3 | 0.075268 | -0.028993 | 0.333506 | 0.476691 |
| fwe | W4 | 1.294981 | 1.022075 | -0.003822 | 0.130625 |
| fwe | W5 | 0.485164 | 0.530033 | -0.014421 | -0.043323 |
| fwe | W6 | 0.398667 | 0.314279 | 0.044520 | 0.111587 |
| fwe | W7 | 0.357375 | 0.223246 | 0.136323 | 0.260920 |
| fwe | W8 | 0.368472 | 0.217970 | -0.003418 | 0.119728 |
| fwe | W9 | 0.239371 | 0.052643 | 0.142567 | 0.345247 |
| fwe | S2 | -0.296443 | 1.184652 | 1.007967 | -0.353344 |
| fwe | S3 | 0.169053 | 0.297311 | 0.269707 | 0.144178 |
| fwe | S4 | 0.909501 | 2.728301 | 1.035943 | 0.042736 |
| nem | W1 | -0.225489 | -0.519324 | 0.118515 | 0.802257 |
| nem | W2 | -0.158909 | -0.109273 | 1.149663 | 1.029871 |
| nem | W3 | 0.498893 | 0.144377 | 1.304653 | 2.018610 |
| nem | W4 | 0.058023 | 0.149579 | 0.052429 | -0.031389 |
| nem | W5 | -0.039524 | -0.066119 | 0.044113 | 0.073847 |
| nem | W6 | 0.076071 | 0.011021 | 0.081790 | 0.151393 |
| nem | W7 | 0.169799 | 0.200485 | 0.209834 | 0.178909 |
| nem | W8 | 0.275450 | 0.349028 | 0.213056 | 0.146895 |
| nem | W9 | 0.309117 | 0.152429 | 0.128916 | 0.282408 |
| nem | S2 | 0.059622 | 0.266550 | 1.657121 | 1.223004 |
| nem | S3 | -0.478140 | -0.467518 | 0.307518 | 0.281437 |
| nem | S4 | -0.104946 | 0.274928 | 1.772607 | 0.946488 |

Full absolute/relative estimates, intervals, damages, correlations and every prediction clause are in `decision_depth.json`.

Spec defines contrasts in nats and additionally requests relative values. Both clause evaluations are reported explicitly, without choosing a scale after observing results.

Predictions (nats): `{"A1": false, "A2": false, "A3": false, "A4": false}`.

Predictions (relative): `{"A1": false, "A2": false, "A3": false, "A4": false}`.

Scientific flags: 0. Failed checks retain their original values; remaining registered work continues.

Deviations: ["Breadth completed before Depth A/B because Depth spec was unavailable; historical ordering deviation preserved."].

Ledger: see `ledger.json`; estimates and elapsed time are report-only.



## Independent completion evidence — 01Nov15:13 SGT

DT4 and all12 Part B fits complete/full762 LR updates/exact eight-rank data order/roundtrip0/no scientific flags. All176 frozen files match a9cbd433. Training ended14:53:43SGT; native supervisor finished14:54:54; guard released and fresh burn progress530→540/payload588.49→599.59GiB independently verified15:02.

205 downloaded files and39 per-sequence arrays have exact SHA matches; all13 result files match terminal receipts. Independent CPU replay504checksPASS, maximum difference5.218048215738236e-15. Recomputed E/A_init/relative values/labels/B1/B2; B1 true and B2 false, preserved. [Download proof](partb_verified/download_verification.json), [replay](partb_verified/independent_replay.json), [native handoff](partb_terminal/verification.json). Original bulk files are Git-ignored under artifacts/partb; retain them separately for backup.

A+B report posted in this thread before assembling C. Breadth-before-Depth ordering deviation retained. A historical local health-summary A_init formula was corrected transparently; raw scores and production formulas did not change ([record](partb_health_1302/A_init_correction.json)).

C recovery implementation: clean full-state copy outside checkpoint rotation, nonfinite attempts synchronized before backward, three bounded retry attempts, exact completed-arm/score reuse and native existing-root continuation.20CPU testsPASS including two CPU/Gloo ranks; no actual eight-B200/NCCL resume test. No C launch/productionDT6 claimed in this report snapshot. C39nodeh is an unmeasured report-only estimate.

Remaining: productionDT6, three C12.8B fits/per-run notes, downstream/pairedCI/independent replay/finalreport/nativehandoff.


## wc69 recovery update — 04Nov2026 SGT

X72 rerun complete: final NLL 2.817733402529562, PPL 16.73886736753704; 48,828 updates, roundtrip0, no run scientific flags. All23 result files and3 final-set arrays downloaded with exact SHA; independent CPU replay164PASS. [X72 report and backup evidence](wc69_x72_20261104/REPORT.md). P300 is active; P1T, downstream, pairedCI, final replay/report and native handoff remain. Earlier C scalar results are historical and are not mixed with this rerun. Historical deviations and CPU-only resume limitation remain.
