# A2 Depth — đang chạy

Spec: `docs/A2_depth_spec.md` v1.1, SHA-256 `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`.

Part A bắt đầu 01/11/2026 10:03:15 SGT. Snapshot 10:32:18 SGT xác nhận 8 B200 hoạt động, đúng coordinator/supervisor, 163 file nguồn giữ nguyên. Metadata ghi nhận 94/156 arrays, mỗi array 2461 sequences / 10077795 targets; full saved-array replay còn chờ export.

- 33 kiểm tra chuẩn bị PASS; 118 kiểm tra worker PASS; cả 8 DT1 so với Q đã có sai khác đúng bằng 0. Bốn parent NEM còn chưa bắt đầu tại snapshot này.
- 5 kiểm thử CPU Part A PASS, gồm phép pruning native, paired readout và tiếp tục sau lỗi.
- DT4/Part B đã chuẩn bị adapter/coordinator/readout/launcher và công cụ lập plan từ bàn giao thực tế; 7 kiểm thử CPU PASS. Chưa lập plan chạy thật hoặc khởi chạy B; cần Part A terminal và note kết quả thực tế.
- Mốc Stage 1/1b chính xác: `resources/a2_depth_stage1_references_20261101.json`.
- Ước lượng spec: khoảng 42 node-giờ cho toàn bộ A/B/C/downstream; chưa hiệu chỉnh bằng throughput thực tế. Không có hard timeout/deadline.
- Worker nhắc cùng thread PID1283867, chu kỳ 30 phút, lần kế tiếp 11:00:55 SGT; đã xác nhận actual delivery vào cùng thread, có delivery_ack_depth.json.

Tiến độ đo được khoảng 600 giây/4 phép đánh giá mỗi lane. Với hai đợt parent trên các lane 0–3, ETA Part A khoảng **01/11 11:10–11:25 SGT**, gồm readout, còn tạm tính; ước lượng 0,6 node-giờ ban đầu thấp. Ước lượng toàn bộ khoảng 42–43 node-giờ vẫn chưa được đo trên C.

Nguồn đang chạy: `1be9ccb33fb146e6e6712aff140e8c3346f0fbdb`. Receipt và bằng chứng startup: [startup_1005/verification.json](startup_1005/verification.json). Tiến độ mới nhất: [health_1032/verification.json](health_1032/verification.json).

Sai lệch giữ nguyên: Breadth đã hoàn tất trước Depth A/B do spec Depth chưa có lúc đó. Spec hiện đã được cung cấp; không còn yêu cầu user bổ sung.

Rà soát code theo yêu cầu user (01/11): **11/11 kiểm thử CPU PASS**, không skip. Bằng chứng và hash: [code_review_20261101.json](code_review_20261101.json).

- Phép cắt layer native/restoration, công thức paired contrasts, KL/gradients, optimizer/scheduler kế thừa và thứ tự DT4 + 12 fits phù hợp với các kiểm thử đã chạy.
- Đã sửa readout để gom cờ lỗi preparation/coordinator; kiểm tra số sequences, targets và mean thay vì chỉ đếm filename; đánh dấu thiếu coverage nhưng vẫn tính phần dữ liệu hữu dụng và giữ cờ lỗi.
- Đã bổ sung cờ lỗi tổng hợp từ mọi rank vào result và per-sequence outputs của short heals. Các sửa này nằm trong code cho bước tiếp theo/readout độc lập; không sửa source Part A đã đóng băng đang chạy.
- Helper full-state checkpoint/resume đã kiểm thử bằng native Trainer trên CPU: weights/optimizer/scheduler, mẫu RNG Python/NumPy/Torch, thứ tự mẫu và LR trùng khớp chạy liền mạch. Lưu model FP32 + optimizer/scheduler + RNG + Trainer state và trace từng rank; checkpoint cuối khoa học vẫn BF16. Mặc định giữ 2 training checkpoints mới nhất trong root C mới, không đụng checkpoint cũ của dự án.
- Kiểm thử bổ sung 01/11 10:40 SGT: native full-state resume **2 rank CPU/Gloo PASS**, so sánh bitwise weights và trạng thái optimizer/scheduler/RNG, LR và thứ tự mẫu với chạy liên tục. Hai lần fixture lỗi `cpu:0` trước đó được giữ trong bằng chứng; sửa bằng `ACCELERATE_TORCH_DEVICE=cpu` chỉ trong subprocess kiểm thử CPU, không sửa numerical core/GPU.
- Launcher DT4/B và ghép A+B đã CPU validated. DT6 có code kiểm tra full-stream hash lịch sử để xác nhận prefix, cộng kiểm tra độc lập view/order của PackedRows; fixture phát hiện được thay đổi ngoài prefix. Chưa chạy DT6 trên stream thật, không tuyên bố đã có historical prefix-only hash.
- Bằng chứng: [cpu_preparation_1040/partb_validation.json](cpu_preparation_1040/partb_validation.json), [cpu_preparation_1040/resume_validation.json](cpu_preparation_1040/resume_validation.json); các hash nguồn/kiểm thử được đối chiếu lại trước commit.
- Giới hạn: chưa tích hợp resume vào C hoặc xác minh resume 8 GPU. C adapter/coordinator/downstream chưa triển khai đầy đủ hay queued. Không tuyên bố toàn pipeline đã được chứng minh đúng.

Còn lại: Part A và bảng A1–A4; DT4; 12 lượt heal Part B và báo cáo A+B; DT6; 3 lượt heal 12.8B có checkpoint đầy đủ để resume; downstream; replay độc lập, báo cáo cuối và native burn handoff. Scientific/time checks ghi nhận và tiếp tục; không chuyển FAIL thành PASS.
