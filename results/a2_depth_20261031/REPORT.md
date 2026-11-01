# A2 Depth — Part A hoàn tất; B/C còn tiếp tục

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

Remaining: DT4, Part B and its report, DT6, Part C, downstream, independent replay and native handoff.

Xác minh local: 222 file tải đủ và đúng SHA; 156 arrays; replay độc lập 1.938 phép đối chiếu, max sai khác 3.375077994860476e-14. Readout sửa kiểm tra coverage giữ nguyên numerics/clauses, 0 cờ khoa học.

Part A kết thúc 01/11 11:06:32 SGT; native handoff 11:07:38 SGT; fresh burn 11:31 cycles1660→1680, payload1843.18→1865.39 GiB. Runtime job3796.785sec ≈1.0547nodeh (ước lượng ban đầu0.6h).

Bằng chứng: [replay độc lập](parta_verified/independent_replay.json), [audit coverage](parta_verified/readout_audit.json), [bảng và báo cáo Part A](parta_verified/REPORT.md), [terminal/handoff](parta_terminal/verification.json). Quyết định đầy đủ lưu lossless tại parta_verified/decision_depth.json.gz; originals gồm156arrays trong artifacts/, cần backup riêng vì Gitignored.

DT4/B launcher đã CPUvalidated7tests; C worker/resume/coordinator/readout5testsPASS, thêm2rankCPUresumePASS. C launcher/actualhandoff/downstream vẫn cần tích hợp trước chạy. Chưa có B/C GPU result; không tuyên bố toàn bộDepth hoàn tất.
