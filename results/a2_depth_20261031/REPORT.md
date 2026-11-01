# A2 Depth — Part A hoàn tất; DT4/B đang chạy

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


## Tiến độ DT4/B và chuẩn bị C — 01Nov11:53 SGT

DT4/B bắt đầu 11:46:14 SGT; snapshot 11:48:52 xác nhận DT4 70/762 updates, 8 GPU 98–99%, 43 preparation checks và 32 worker checks ban đầu PASS. Source `a9cbd433`/176 files bất biến; đúng burn được thu hồi và quyền sở hữu đầy đủ được xác minh. Receipt: [partb_startup_1148/verification.json](partb_startup_1148/verification.json). DT4 chưa có final NLL; chưa có kết quả B cuối. ETA DT4/B 14:30–15:30 SGT, sơ bộ.

C worker/resume, plan assembler, launcher, 3-student batch-2 downstream, exact parent-score reuse, paired 10k seed 0/fresh per-item replay và ghép report đã qua 10 CPU tests: [validation](cpu_preparation_1150/validation.json). Chưa lập actual C handoff plan, chưa launch C; cần B terminal và A+B report trước. Native 8-GPU resume chưa được xác minh thực nghiệm.

Còn lại: DT4/B hoàn tất và replay/report A+B; DT6 production, C ba full schedules, downstream/replay/final report/native handoff. Breadth-before-Depth deviation giữ nguyên.


## DT4 hoàn tất, B đang chạy — snapshot 01Nov12:02 SGT

DT4 đã chạy đúng một lần, đủ 762 updates; final NLL **2.9429481391618784**, so với Stage1 reference 2.942554079, chênh **+0.0003940601618781514**, PASS ngưỡng 0.002. Tuning NLL 2.9310933019668948, save/reload difference 0; không có flags. Thời gian training 734.423s, toàn fit 786.049s. Đây là kiểm tra từ receipt/result summary; replay toàn bộ saved arrays sẽ thực hiện khi tải kết quả B.

Lượt B đầu tiên DCLM/P300/W9 đang chạy (1/762 tại snapshot), tám GPU 99%; 128 worker checks và 43 preparation checks PASS, 176 source files không đổi, full ownership được xác minh. Bằng chứng: [verification](partb_health_1202/verification.json). ETA B khoảng14:40–15:30 SGT, sơ bộ từ DT4; C 39h vẫn là estimate chưa đo.

Còn lại: B12/replay/A+B note, DT6 production, C/downstream/final report/native handoff. Deviation Breadth-before-Depth giữ nguyên.


## Part B — snapshot 01Nov12:32 SGT

2/12 B fits hoàn tất: DCLM/W9 P300 final NLL 3.0268649928550975 (damage0.28907196064241303); P1T final3.009014625735566 (damage0.3221056339716317). E_W9=+0.033033673329218693 tính từ summaries; full saved-array replay còn chờ. Cả hai đủ762 LR updates, roundtrip0, không flags. DT4 giữ nguyên PASS.

DCLM/W9 X72 đang chạy70/762,8GPU98–100%;304worker checks/43prepPASS,176sourcefiles không đổi/full ownership verified. [Receipt và verification](partb_health_1232/verification.json). Mỗi fit gồm startup hiện khoảng14.5phút; dự kiến B14:50–15:15SGT, sơ bộ. C39h vẫn chưa đo.

Còn lại:10fits B/replay/A+B note,DT6/C/downstream/finalreport/nativehandoff; deviation Breadth–Depth giữ nguyên.


## C recovery implementation — 01Nov2026 (CPU validation only)

User-authorized review fixes completed before C launch. The C trainer flags and synchronizes a nonfinite loss across ranks, leaves that failed attempt before backward, and the coordinator retries from clean full-state evidence. Finite-loss KD/optimizer/LR/sample order are unchanged. All failure flags remain in the report.

- Native rotation keeps two ordinary checkpoints; one additional copied clean checkpoint is kept under trainer/clean-state, outside checkpoint-* rotation. A replacement copy is staged before the older clean copy is removed. Nonfinite audit/state candidates are rejected.
- Same-step transient failures receive up to three attempts with10/20s backoff; each uses an available rendezvous port. Physical GPUs must become free; no unknown processes are killed.
- `launch_a2_depth_partc.py --resume --output EXISTING_ROOT --plan ORIGINAL_FROZEN_PLAN --code-commit ORIGINAL_COMMIT --session UNIQUE_SESSION --recovery-handoff FRESH_JSON` resumes the existing scientific experiment under a new native supervisor. Fresh handoff JSON is either `{"mode":"burn","predecessor":{...exact current locks/start identities...}}` or `{"mode":"free"}`. Free mode independently verifies original host/GPU UUIDs, no GPU workers, no held guard and no live prior supervisor/coordinator. Existing source/config/plan must match their original hashes. Historical supervisor/burn receipts are preserved under their original paths; new receipts are indexed in continuations.json.
- Completed arms validate result/recipe/order/rank checks/final model hashes before reuse. Downstream can reenter; sealed scores and matching replays are reused, partial files remain in earlier attempt directories.

20CPU tests PASS, including exact two-rank Gloo resume (weights/optimizer/scheduler/RNG/LR/order) and NaN on one rank unwinding both ranks before backward. [Evidence](cpu_recovery_20261101/validation.json). Initial test-fixture failures are retained alongside the passing log. No actual eight-B200/NCCL resume test or C launch is claimed.

Remaining: B/replay/A+B note,production DT6/C/downstream/finalreport/nativehandoff. Breadth ordering deviation unchanged.


## Part B — snapshot 01Nov13:02 SGT

4/12 B fits complete: all three DCLM/W9 arms and FWE/W9 P300. New final NLL: DCLM/X72/W9 3.1039990815860548 (damage 0.19542378547818462); FWE/P300/W9 3.1170931009343033 (damage 0.25646177507748824). Each completed fit has 762 LR updates, roundtrip difference 0 and no flags. Prior results and DT4 are unchanged. Full saved-array replay remains pending.

FWE/P1T/W9 is active at 140/762 updates; all eight GPUs active (86–100%). All 176 frozen source files unchanged, full process ownership verified, 43 preparation and 480 worker checks PASS. Complete scans of all six existing training logs found zero nonfinite flags. [Evidence](partb_health_1302/verification.json). B ETA 14:50–15:15 SGT remains provisional; C's 39h estimate remains unmeasured.

Remaining: eight B fits including the active one, replay/A+B note, DT6/C/downstream/finalreport/nativehandoff. Breadth ordering deviation and CPU-only C resume limitation retained.
