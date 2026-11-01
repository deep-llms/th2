# A2 Depth — đang chạy

Spec: `docs/A2_depth_spec.md` v1.1, SHA-256 `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`.

Part A bắt đầu 01/11/2026 10:03:15 SGT. Snapshot 10:15:04 SGT xác nhận 8 B200 hoạt động (69–78%), đúng coordinator/supervisor, 163 file nguồn giữ nguyên. Metadata ghi nhận 32/156 arrays (8 parent + 24 cấu hình W1–W3), mỗi array 2461 sequences / 10077795 targets; full saved-array replay còn chờ export.

- 33 kiểm tra chuẩn bị PASS; 48 kiểm tra worker PASS (DT3, parent-vs-Q, DT2 tới W4). DT1 còn chờ W8.
- 5 kiểm thử CPU Part A PASS, gồm phép pruning native, paired readout và tiếp tục sau lỗi.
- DT4/Part B đã chuẩn bị adapter/coordinator/readout, 3 kiểm thử CPU PASS; chưa khởi chạy.
- Mốc Stage 1/1b chính xác: `resources/a2_depth_stage1_references_20261101.json`.
- Ước lượng spec: khoảng 42 node-giờ cho toàn bộ A/B/C/downstream; chưa hiệu chỉnh bằng throughput thực tế. Không có hard timeout/deadline.
- Worker nhắc cùng thread PID1283867, chu kỳ 30 phút, lần kế tiếp 10:30:55 SGT; đã xác nhận actual delivery vào cùng thread, có delivery_ack_depth.json.

Tiến độ đo được khoảng 600 giây/4 phép đánh giá mỗi lane. Với hai đợt parent trên các lane 0–3, ETA Part A khoảng **01/11 11:10–11:25 SGT**, gồm readout, còn tạm tính; ước lượng 0,6 node-giờ ban đầu thấp. Ước lượng toàn bộ khoảng 42–43 node-giờ vẫn chưa được đo trên C.

Nguồn đang chạy: `1be9ccb33fb146e6e6712aff140e8c3346f0fbdb`. Receipt và bằng chứng startup: [startup_1005/verification.json](startup_1005/verification.json).

Sai lệch giữ nguyên: Breadth đã hoàn tất trước Depth A/B do spec Depth chưa có lúc đó. Spec hiện đã được cung cấp; không còn yêu cầu user bổ sung.

Rà soát code theo yêu cầu user (01/11): **11/11 kiểm thử CPU PASS**, không skip. Bằng chứng và hash: [code_review_20261101.json](code_review_20261101.json).

- Phép cắt layer native/restoration, công thức paired contrasts, KL/gradients, optimizer/scheduler kế thừa và thứ tự DT4 + 12 fits phù hợp với các kiểm thử đã chạy.
- Đã sửa readout để gom cờ lỗi preparation/coordinator; kiểm tra số sequences, targets và mean thay vì chỉ đếm filename; đánh dấu thiếu coverage nhưng vẫn tính phần dữ liệu hữu dụng và giữ cờ lỗi.
- Đã bổ sung cờ lỗi tổng hợp từ mọi rank vào result và per-sequence outputs của short heals. Các sửa này nằm trong code cho bước tiếp theo/readout độc lập; không sửa source Part A đã đóng băng đang chạy.
- Helper full-state checkpoint/resume đã kiểm thử bằng native Trainer trên CPU: weights/optimizer/scheduler, mẫu RNG Python/NumPy/Torch, thứ tự mẫu và LR trùng khớp chạy liền mạch. Lưu model FP32 + optimizer/scheduler + RNG + Trainer state và trace từng rank; checkpoint cuối khoa học vẫn BF16. Mặc định giữ 2 training checkpoints mới nhất trong root C mới, không đụng checkpoint cũ của dự án.
- Giới hạn: resume mới kiểm thử một rank CPU, chưa tích hợp vào C hoặc xác minh đa rank. DT4/B launcher và ghép báo cáo A+B còn cần hoàn thiện trước handoff; DT6/C/downstream chưa được triển khai đầy đủ hay queued. Không tuyên bố toàn pipeline đã được chứng minh đúng.

Còn lại: Part A và bảng A1–A4; DT4; 12 lượt heal Part B và báo cáo A+B; DT6; 3 lượt heal 12.8B có checkpoint đầy đủ để resume; downstream; replay độc lập, báo cáo cuối và native burn handoff. Scientific/time checks ghi nhận và tiếp tục; không chuyển FAIL thành PASS.
