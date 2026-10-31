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

## Kiểm chứng CPU bổ sung lúc 08:30 SGT

Health a04 vẫn chưa trả receipt; controller gần nhất ghi OK started. Không suy ra trạng thái GPU hiện tại từ việc log chậm.

Đã tính lại paired bootstrap 10.000 lượt, seed 0 trên các file final NLL nhận ở snapshot 07:02. Đây là kết quả tạm từ dữ liệu đã nhận, **unverified** do QT3/QT6; thiếu cal1 sensitivity và GPTQ NEM, chưa đối chiếu decision remote. Không phải kết luận quant cuối.

| Dataset | GPTQ3 contrast | Estimate (nats/token) | 95% CI |
|---|---|---:|---|
| dclm | L_ann | 0.020258419 | [0.019900407, 0.020613748] |
| dclm | L_stab | 0.000445813 | [0.000144404, 0.000741646] |
| dclm | R_short | 0.003595412 | [0.003286911, 0.003899501] |
| fwe | L_ann | 0.010195175 | [0.009876725, 0.010508563] |
| fwe | L_stab | 0.004798875 | [0.004498679, 0.005100016] |
| fwe | R_short | 0.005807322 | [0.005508324, 0.006115979] |

Chi tiết đầy đủ: [partial_paired_nll.json](health_0702/partial_paired_nll.json).

Còn cần: receipt a04, trạng thái hiện tại, 32 NLL chưa nhận, cal1 sensitivity/NEM GPTQ và đối chiếu CI, downstream/replay, final report/native handoff. QT3/QT6 FAIL giữ nguyên.

## Receipt theo dõi lúc 09:00 SGT

09:00 SGT: Health a04 vẫn chưa có receipt; controller gần nhất chỉ ghi OK started (23:32:53 UTC). Chưa có bằng chứng lỗi controller hoặc trạng thái GPU mới. Đã khoảng 2 giờ 56 phút theo đồng hồ kể từ startup 06:04:17, vượt dự toán 2 giờ; đây không phải compute time đã xác minh. Không restart hoặc gửi lệnh trùng. Trạng thái xác minh cuối vẫn là 07:02:06 SGT, cal1 active, 128 NLL đã nhận và kiểm chứng.

Còn cần hoàn tất: receipt a04, xác minh tiến độ và phần kết quả còn lại, CI/downstream/replay, final report và native handoff. QT3/QT6 FAIL giữ nguyên.

09:30 SGT: receipt a04 vẫn chưa về sau khoảng hai giờ. Dropbox truy cập được; status controller không đổi. Trạng thái job sau 07:02 chưa xác định. Không có kết quả mới để công bố; không suy ra job đang chạy/đã xong/bị lỗi. Watcher hiện tại tiếp tục lúc 10:00 SGT.

10:00 SGT: không có receipt/status mới. Health a04 pending khoảng 2,5 giờ; trạng thái job sau 07:02 chưa xác định. Giữ nguyên job, không gửi lại lệnh. Còn thiếu receipt, phần NLL/CI chưa nhận, downstream/replay và báo cáo/bàn giao; giữ QT3/QT6 FAIL.

## Lần kiểm tra receipt gần nhất

31 Oct 13:00 SGT: Dropbox truy cập được; receipt a04 vẫn chưa về sau khoảng 5,5 giờ chờ. Status mới nhất vẫn là 11:39:34 SGT của lần probe đã nhận. Trạng thái quant sau 07:02 chưa xác định; chưa có kết quả mới hoặc ETA đủ căn cứ. Watcher hiện tại PID 1095005 tiếp tục lúc 13:30 SGT. Không có remote write, gửi lại lệnh hay relaunch.

Còn thiếu: receipt, 32 NLL chưa nhận, đối chiếu CI, downstream/replay, final report và native handoff. QT3/QT6 FAIL giữ nguyên.

## Probe nhận log 11:39 SGT

Theo yêu cầu user, đã push commit mới `4f6f3ce`; Dropbox nhận log lúc 11:39:32 SGT, controller xác nhận started + synced. Đường commit mới → thực thi → trả log hoạt động. Probe lỗi ở lệnh `git rev-parse HEAD` vì thư mục sync trên node không có `.git`; không tác động GPU/job quant. Receipt a04 cũ vẫn chưa nhận, nguyên nhân chưa xác định. [Bằng chứng](dropbox_probe_1139/receipt.json).

## Xác minh máy và trạng thái thực thi — 13:09 SGT

Receipt [node_identity_1309/receipt.json](node_identity_1309/receipt.json) xác nhận cả 8 UUID GPU khớp chính xác snapshot 07:02; hostname cũ, boot từ 22/09/2026 14:43:49 SGT, dữ liệu/model/output và môi trường cũ còn tồn tại. Đây không phải máy B200 mới tinh.

Job quant đã kết thúc lúc **31/10 08:33:05 SGT**, exit 0; supervisor ghi bàn giao burn lúc **08:34:16 SGT**. Đã kiểm chứng chuỗi SHA của run complete → quant complete → result complete. Quan sát được **80 final + 80 own-corpus NLL**; result complete báo **16 downstream**, coverage_complete=true và **2.47948 node-hours**. Đây là bằng chứng hoàn tất thực thi; chưa thay thế kiểm chứng độc lập toàn bộ kết quả tải về. Các GPU hiện mang PID 740109–740116 khớp danh sách burn trong supervisor; log burn tiến triển 18840→18850. Cần kiểm tra đầy đủ danh tính/ownership hiện tại trước khi chốt terminal verification.

QT3/QT6 FAIL vẫn giữ nguyên (complete ghi 9610 flagged_checks). Health a04 cũ chưa nhận, nguyên nhân chưa xác định; các mô tả UNKNOWN ở trên là trạng thái lịch sử trước receipt mới. Còn cần lấy phần NLL/cal1/NEM, decision/CI/downstream/replay/REPORT để kiểm chứng và chốt báo cáo cùng terminal handoff.
