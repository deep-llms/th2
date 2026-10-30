# A2 quant — đang chạy

**Cập nhật 07:02:06 SGT, lấy về 07:16:** baseline/pre-S1 + RTN cả 12 parent hoàn tất; GPTQ3/4 DCLM/FWE8 hoàn tất; cal1 đang chạy cả 8 GPU (70–83%). QT5 full-parent repeat PASS. 128 file NLL đã được tính lại mean độc lập từ từng sequence. QT3/QT6 vẫn FAIL. Chưa hoàn tất CI/downstream/replay/report. Các bảng bên dưới là snapshot 06:31; số mới nằm trong `health_0702/nll_verified.json`.

Receipt mới nhất đã xác minh: **31 Oct 2026 06:31:50 SGT**, lấy về 07:00 SGT. Source `422bb7f`, 144 file không đổi, ownership đúng. Baseline/pre-S1 12/12 hoàn tất; RTN DCLM/FWE 8/8 hoàn tất cả 3/4 bit; NEM RTN4 xong, RTN3 đang chạy. Có 88/160 file NLL được báo trong receipt. Chưa có CI/label quant cuối.

QT1: 12 parent khớp lịch sử, sai khác lớn nhất `4.440892098500626e-16`. Các số dưới đây lấy từ worker receipts; replay độc lập từ từng sequence đang chờ health a03.

| Dataset | Parent | Parent NLL | pre-S1 NLL | RTN4 NLL | RTN3 NLL |
|---|---|---:|---:|---:|---:|
| dclm | P1T | 2.686908991764 | 4.004721485197 | 2.698595929261 | 2.763839095581 |
| dclm | P300 | 2.737793032213 | 3.763775143788 | 2.748440407820 | 2.803538977965 |
| dclm | S188 | 2.886706698044 | 4.012656808595 | 2.897208018378 | 2.934568827327 |
| dclm | S72 | 2.908575296108 | 3.854544222001 | 2.917861542017 | 2.963282927930 |
| fwe | P1T | 2.808707333602 | 4.124982157487 | 2.819737037068 | 2.865632341260 |
| fwe | P300 | 2.860631325857 | 3.822488626156 | 2.871162211870 | 2.914961756164 |
| fwe | S188 | 3.023807591626 | 4.199338768225 | 3.032904102262 | 3.070687022585 |
| fwe | S72 | 3.042152001946 | 4.007307971499 | 3.049441876500 | 3.081262267948 |
| nem | P1T | 2.723382862027 | 4.116536224985 | 2.736076477795 | Chưa có |
| nem | P300 | 2.774250937160 | 3.866534354761 | 2.785776165979 | Chưa có |
| nem | S188 | 2.926708960676 | 4.141426296314 | 2.934729832262 | Chưa có |
| nem | S72 | 2.952463783417 | 3.852902816181 | 2.960013122666 | Chưa có |

Cờ giữ nguyên: QT6 cal0/cal1 chung một document ranh giới; không trùng tuning/final. QT3 refit-grid idempotence FAIL ở initial checks và RTN matrices; kết quả ảnh hưởng là unverified. QT4 identity-H PASS. Job tiếp tục theo policy.

Pooled downstream: [JSON](pooled_downstream.json), bảng post hoc tại [kết quả Stage1b](../../docs/A2_STAGE1B_FWE_RESULTS_20261030.md).

Còn cần hoàn tất: RTN NEM, GPTQ/cal1, CI/labels, downstream, replay độc lập, REPORT.md/decision và native terminal handoff. Health a03 đang chờ; watcher cùng thread PID1095005 tiếp tục 07:30 SGT, không deadline.
