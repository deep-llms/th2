# Trả lời xác nhận Stage1b trước interim

Nội dung trả lời ngày 30/10/2026, trước interim; trạng thái dưới đây là tại thời điểm trả lời.

P1T 3.2B vẫn tiếp tục chạy. Tôi đã lưu [audit đầy đủ từng file](A2_STAGE1B_CODE_AUDIT_20261030.md), gồm phân loại numerical path và toàn bộ diff Git.

**1a. Training code**

Trong snapshot frozen **103 → 124 file**: **99 giữ nguyên, 4 sửa, 21 thêm, không xóa**.

Bốn file sửa:

| File | Thay đổi và ảnh hưởng numerical |
|---|---|
| `scripts/a2_stage1_data.py` | Tham số hóa corpus và kiểm tra receipt nguồn. Chạm đường dữ liệu; thuật toán shuffle/packing giữ nguyên. |
| `eval/a2_checkpoint.py` | Thêm dispatch xác minh checkpoint schema2; chạm đường load/identity evaluation. |
| `eval/a2_run.py` | Tách coordinator dùng chung, tham số hóa validator/gates/finalizer; chạm orchestration evaluation. |
| `scripts/launch_a2_downstream.py` | Cho phép freeze thêm file; không đổi phép tính số. |

**21 file thêm được liệt kê riêng từng file trong audit**, gồm adapter FWE training/data, coordinator hai phase, downstream 4+2, bootstrap, launchers và spec/locks.

Đối chiếu các thành phần chính:

| Thành phần | Kết quả |
|---|---|
| Model forward, layer removal, KL | Implementation cũ giữ nguyên byte; adapter mới chọn FWE model/teacher. |
| Optimizer, LR schedule, precision | Implementation và Accelerate config giữ nguyên byte. |
| Data packing/order | Có thay đổi chọn corpus/admission; shuffle seed2030 và packing giữ nguyên thuật toán. |
| Evaluation/scoring | Có adapter, checkpoint identity và thống kê mới; không thể gọi toàn bộ thay đổi là “chỉ docs”. |

**1b. Có**, cả 8 grid runs, P300 3.2B và P1T 3.2B đang chạy dùng frozen snapshot **`7dd9da0`**.

Một chi tiết provenance cần làm rõ: `source-manifest.code_commit` là `7dd9da0`, nhưng `result.provenance.code_commit` giữ **`9f5601d` của preflight**. Launcher kiểm tra **27 core-file SHA giống nhau** trước khi kế thừa provenance; tôi đã kiểm tra lại chúng và cả 9 result đã xuất bản. P1T chưa có result cuối.

**1c. T17 đã chạy lại và PASS trước FWE.**

- Hoàn tất **04:04:53 SGT**, trước training bắt đầu **04:19:10 SGT**.
- Đúng run `P0-nem-P300-S1-0.05B-lr3e-4`, CulturaX stream gốc, 190 update, 64×4096 token/update; đã kiểm tra full LR schedule.
- Original NLL: **3.061875752041902**.
- Rerun NLL: **3.0617311959690636**.
- Chênh lệch: **−0.000144556073 nats**, trong ±0.002.

Vì T17 đã chạy trước FWE và core SHA không đổi sau preflight, điều kiện phải chạy bổ sung sau P1T không phát sinh.

**2. Downstream sẽ không dùng nguyên commit `13686ed`.**

Có adapter FWE 4+2 và thay đổi identity/bootstrap. Các file scorer/request/replay chính giống byte với `13686ed`, nhưng đó chưa phải bằng chứng GPU equivalence.

Đã ghi thành bước bắt buộc: **sau P1T kết thúc và xác minh handoff, re-score DCLM P300 parent trước downstream FWE/Part2**, dùng đúng identity, canonical requests và precision của Stage1; đối chiếu scores cùng metrics. Bước này **chưa chạy**.

**3. Hai spec hash đều đúng trong receipts đã kiểm tra:**

- v1.1: `b2121a4f2859f99db421273082c6be035dc5e833af6d1d071f5ed4cd381ed483`
- v3.3: `9ebe0d0a377f0107f7058bf5e4899b5380cbe5f4f518b09091ac533d2c9f254e`

Đã xác minh trong preflight/tests, inputs Part1 và cả 9 result hiện có; sẽ kiểm tra tiếp receipt cuối P1T.
