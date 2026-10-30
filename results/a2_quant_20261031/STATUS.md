# A2 quant — đang chạy

Xác minh 31 Oct 2026 06:06:40 SGT: job bắt đầu 06:04:17, tám worker baseline DCLM/FWE đang chạy trên 8 GPU. Source `422bb7f`, 144 file không đổi; ownership và native handoff đã xác minh. Đây chưa phải kết quả quant cuối.

Pooled downstream đã có trong [pooled_downstream.json](pooled_downstream.json), cùng bảng post hoc tại [kết quả Stage1b](../../docs/A2_STAGE1B_FWE_RESULTS_20261030.md).

Cờ cần giữ: QT6 cal0/cal1 chung một document ranh giới; không trùng tuning/final. QT3 refit-grid idempotence initial check FAIL ở 3/4 bit; cardinality và QT4 identity-H PASS. Job tiếp tục; kết quả bị ảnh hưởng phải ghi unverified.

Còn cần hoàn tất: các phép đo quant/160 NLL files, paired CI, downstream, replay độc lập, REPORT.md/decision và native terminal handoff. Watcher cùng thread: PID1095005, kỳ tiếp theo 06:30 SGT, không deadline.
