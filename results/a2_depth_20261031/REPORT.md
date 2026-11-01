# A2 Depth — đang chạy

Spec: `docs/A2_depth_spec.md` v1.1, SHA-256 `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`.

Part A bắt đầu 01/11/2026 10:03:15 SGT. Snapshot 10:05:53 SGT xác nhận 8 B200 hoạt động (70–82%), đúng coordinator/supervisor, 163 file nguồn giữ nguyên. Chưa có kết quả NLL hoàn chỉnh tại snapshot này; không coi là đã hoàn tất 144 cấu hình.

- 33 kiểm tra chuẩn bị PASS; 8 worker đầu tiên xác nhận DT3 PASS. DT1/DT2 còn chờ kết quả.
- 5 kiểm thử CPU Part A PASS, gồm phép pruning native, paired readout và tiếp tục sau lỗi.
- DT4/Part B đã chuẩn bị adapter/coordinator/readout, 3 kiểm thử CPU PASS; chưa khởi chạy.
- Mốc Stage 1/1b chính xác: `resources/a2_depth_stage1_references_20261101.json`.
- Ước lượng spec: khoảng 42 node-giờ cho toàn bộ A/B/C/downstream; chưa hiệu chỉnh bằng throughput thực tế. Không có hard timeout/deadline.
- Worker nhắc cùng thread PID1283867, chu kỳ 30 phút, lần kế tiếp 10:30:55 SGT; queue đã chấp nhận lần nhắc đầu tiên, chưa xác nhận actual delivery.

Nguồn đang chạy: `1be9ccb33fb146e6e6712aff140e8c3346f0fbdb`. Receipt và bằng chứng startup: [startup_1005/verification.json](startup_1005/verification.json).

Sai lệch giữ nguyên: Breadth đã hoàn tất trước Depth A/B do spec Depth chưa có lúc đó. Spec hiện đã được cung cấp; không còn yêu cầu user bổ sung.

Còn lại: Part A và bảng A1–A4; DT4; 12 lượt heal Part B và báo cáo A+B; DT6; 3 lượt heal 12.8B có checkpoint đầy đủ để resume; downstream; replay độc lập, báo cáo cuối và native burn handoff. Scientific/time checks ghi nhận và tiếp tục; không chuyển FAIL thành PASS.
