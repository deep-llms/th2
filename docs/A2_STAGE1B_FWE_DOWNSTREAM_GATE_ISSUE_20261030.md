# Stage1b FWE downstream numerical gate failure

Ngày 30/10/2026, giờ Singapore (SGT). **Downstream FineWeb-Edu Part1 đã thất bại ở kiểm tra numerical gate của P300 parent trước khi scoring.** Training Part1 đã hoàn tất và được xác minh; DCLM regression cũng PASS. Vấn đề hiện tại là sai khác likelihood giữa batch size 1 và 2 trên checkpoint FWE vượt ngưỡng đang áp dụng.

## Kết quả kiểm tra

Gate dùng 12 request được chọn cố định từ bộ 78.259 request, context tối đa 4.096 token, FP32 parameters với BF16 CUDA autocast. Production batch size là 2.

| Kiểm tra | Giá trị đo được | Ngưỡng | Kết quả |
|---|---:|---:|---|
| Scorer batch1 so với cross-entropy tính độc lập | 0.0 nats/token | ≤0.00002 | PASS |
| Batch1 so với batch2, sai lệch tuyệt đối lớn nhất/token | 0.011164188385009766 | ≤0.01 | FAIL |
| Batch1 so với production batch2 | 0.011164188385009766 | ≤0.01 | FAIL |
| Greedy mismatches ở cả ba phép so sánh | 0 | 0 | PASS |
| Request bị truncation | 0 | Số lượng phải nhất quán giữa các phép tính | PASS |

Sai lệch vượt ngưỡng **0.001164188385009766 nats/token**, khoảng **11,64%**. Đây là mức sai khác giữa hai cách batching, không phải mức tăng NLL của training hay sai lệch accuracy downstream. Hai dòng batch2 và production batch2 kiểm tra cùng batch size; không phải hai failure độc lập.

[Receipt gate gốc](results/a2_stage1b_fwe_20261030/downstream_part1_gate_failure/supervised/run/downstream/P300-parent-check.json) và [mã kiểm tra](../eval/a2_check.py).

## Vì sao DCLM regression PASS nhưng FWE gate FAIL

Regression trước đó re-score **DCLM P300 parent** với đúng checkpoint, request, batch2 và precision của Stage1. Cả 9 primary task accuracies khớp tuyệt đối; max absolute per-request LL difference = **0.0** trên 78.259 request; score file giống hệt byte.

Gate hiện tại kiểm tra **FWE P300 parent**, đồng thời so batch1 với batch2. Checkpoint và phép so sánh khác nhau, nên regression PASS không bảo đảm gate FWE sẽ PASS. Source FWE downstream giữ nguyên 66 file evaluation/model đã dùng trong regression.

[Đối chiếu regression](A2_STAGE1B_FOLLOWUP_REPLY_20261030.md) và [receipt regression](results/a2_stage1b_fwe_20261030/regression_final/verification.json).

## Điều đã biết và điều chưa xác định

Manual-reference error bằng 0 cho thấy scorer batch1 khớp phép tính cross-entropy độc lập trên các request được kiểm tra. Không có greedy mismatch. Tuy nhiên, hai kết quả này không chứng minh accuracy toàn bộ benchmark không bị ảnh hưởng, vì scoring FWE chưa chạy.

**Chưa xác định nguyên nhân gốc.** BF16 batching có thể chọn các GEMM kernel khác nhau; đây là một giả thuyết phù hợp với thiết kế gate, chưa phải chẩn đoán được xác minh cho failure này. Receipt hiện chỉ lưu maximum error và danh sách 12 request, không lưu sai lệch từng request. Vì vậy chưa xác định được request nào tạo maximum, cũng chưa có bằng chứng để quy lỗi cho padding, độ dài chuỗi hay một kernel cụ thể.

Không có bằng chứng trong receipt này về controller/AWS/auth failure. Lỗi được ghi rõ là `A2 downstream numerical gate failed`; job exit code 1.

## Diễn biến và bàn giao GPU

| Mốc SGT ngày 30/10 | Sự kiện |
|---|---|
| 12:43:17 | Part1 training hoàn tất: 8 grid fits và 2 arms 3.2B |
| 13:33:21 | Receipt regression xác minh accuracy khớp và LL max0; được lấy về lúc14:00 |
| 14:03:43 | FWE downstream bắt đầu sau reclaim đúng burn workers và kiểm tra cả8GPU trống |
| 14:04:31 | Job thất bại numerical gate; chưa scoring |
| 14:05:36 | Supervisor hoàn tất khôi phục burn |
| 14:31:32 | Health xác minh guard released và8burnworkers đang tiến triển; receipt được lấy về lúc15:00 |

Burn workers được xác minh tại snapshot là 723690–723697; collective cycles tăng1780→1790 và payload1976.43→1987.53GiB. Đây là bằng chứng bàn giao tại thời điểm snapshot, không phải khẳng định trạng thái GPU ở mọi thời điểm sau đó. [Verification](results/a2_stage1b_fwe_20261030/downstream_part1_gate_failure/verification.json).

## Ảnh hưởng đến kết quả và tiến độ

- **Part1 training vẫn hợp lệ:** common LR và own-best LR của P300/P1T đều1e-3; E(0.2B)=+0.026224221566386152, E(3.2B)=+0.023461150403907727. Nhãn NLL là **persists**, đối chiếu DCLM là **replicated**.
- **Downstream chưa có kết quả:** chưa có score FWE cho bất kỳ model nào, chưa có paired CI; gate của student chưa được chạy. Không điền zero hoặc dùng điểm DCLM thay thế.
- **Part2 chưa bắt đầu:** X72, các điểm0.8B, A_init và báo cáo cuối còn thiếu. Full Stage1b chưa hoàn tất.
- Đã gửi [REPORT_interim.md](REPORT_interim.md) và [decision JSON](decision_stage1_fwe_interim.json) với trạng thái một phần. ETA cũ20–21SGT không còn là cam kết; trước bước tiếp theo phải tính lại ledger, mốc báo cáo20nodeh, hard cap24nodeh và deadline31Oct04:06:27SGT.

## Quyết định đang cần

Ngoại lệ đã được người dùng cho phép là: nếu **DCLM regression accuracy FAIL**, giữ FWE downstream, gửi interim thiếu downstream và tiếp tục Part2 heals. Thực tế regression PASS, còn **FWE numerical gate FAIL**; ngoại lệ đó chưa bao quát rõ trường hợp này.

[Spec v3.3](A2_stage1_spec.md) quy định: “If something cannot be done as written, stop and report; do not improvise.” Vì vậy chưa tự mở rộng ngoại lệ, đổi batch/precision, nới threshold hoặc retry.

Đề xuất quyết định cụ thể: **cho phép tiếp tục các Part2 heals đã đăng ký trong ngân sách, đồng thời giữ FWE downstream và ghi rõ phần còn thiếu**. Nếu được chấp thuận, cần bổ sung nhánh admission gắn với receipt failure, interim đã gửi và native handoff đã xác minh; không giả lập downstream complete. Nếu muốn điều tra numerical gate bằng GPU trước, cần xác định phép đo và ngân sách riêng cho việc đó. Tài liệu này không tự cấp quyền cho phương án nào.

## Provenance và bằng chứng

- FWE downstream execution source: `05b0d0a944b4f01e947e0ea50fea62a01510da54`,122 frozen files.
- Regression execution source: `8b78d9e1051407a45be6676820bc57ac87a872e4`;66 eval/model hashes trùng FWE downstream.
- Part1 training execution: `7dd9da088ba5af581c295775819319ce54b2025c`; result provenance giữ preflight `9f5601d97ddf2940b32ba5e17de8d21b7c7fe1c0`,27core hashes trùng. Published results không sửa.
- Raw health SHA-256: `5005f261292a74d572dd099ed19fcc322147bf58ceb25d6b101dcf918c0abe1b`.
- [Archive SHA256SUMS](results/a2_stage1b_fwe_20261030/downstream_part1_gate_failure/SHA256SUMS), [job log](results/a2_stage1b_fwe_20261030/downstream_part1_gate_failure/supervised/run/stage1b-downstream-part1.log), [gate log](results/a2_stage1b_fwe_20261030/downstream_part1_gate_failure/supervised/run/downstream/P300-parent-check.log).


## Quyết định mới của người dùng — 30Oct2026

Người dùng đã cho phép Part2 chạy ngay, độc lập với downstream, và phê duyệt vượt mốc báo cáo20nodeh trong giới hạn24nodeh/deadline31Oct04:06SGT. Downstream sáu model chạy sau heals, theo chính sách chi tiết tại `resources/a2_stage1b_continuation_20261030.json`. Mọi lỗi ở phần chỉ báo cáo không chặn heals. Nội dung chờ quyết định phía trên là lịch sử đã được thay thế.
