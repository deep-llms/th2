# DOT-pool pilot — các điểm cần chốt trước khi chạy

> **Đã được xử lý trong v1.2:** [DOT_pool_pilot_spec_v1.2.md](DOT_pool_pilot_spec_v1.2.md), SHA-256 `a0b41acecbd462db6f9277ebec7173fe242d3493d9fb7f2d77b127d88bc97f05`. Nội dung bên dưới giữ lại lịch sử câu hỏi v1.0, không còn là danh sách chặn triển khai. Câu 1–2 đã chốt; câu 3 dùng một main backward và chấp nhận đóng góp balance loss theo spec, không tách backward riêng. Câu 4 có quy tắc tiếp tục rõ ràng: Tier 1 gửi sau mỗi run; Tier 2–3 giữ trên node và ghi “not yet off-node” đến khi chủ project chỉ định cơ chế truyền lớn. Chưa có cơ chế truyền lớn được chỉ định trong lượt đọc này.

Spec: [DOT_pool_pilot_spec.md](DOT_pool_pilot_spec.md), v1.0, ngày 12/11/2026 SGT.

SHA-256: `ee99688ddb75a1b88755db52174ecb7dc81fad9f848a43f3883b6bf3e5d03997`.

Trạng thái: câu hỏi và đề xuất, chưa phải quyết định đã được duyệt. Tài liệu này không sửa spec và không khởi chạy workload.

Ưu tiên của chủ project: kiểm tra DOT-pool có thắng baseline hay không. Giữ nguyên pilot và các kiểm tra tính đúng đã đăng ký; không thêm ablation hoặc thí nghiệm giải thích cơ chế ở giai đoạn này.

## 1. Công thức sequence-wise balance loss (§2.2)

Spec đã quy định hệ số `α = 1e-4` và phạm vi pool/layer, nhưng chưa định nghĩa đầy đủ công thức và cách tổng hợp qua depth.

**Câu hỏi:** Có đồng ý cách tính cụ thể dưới đây không, hay muốn một công thức khác?

Đề xuất cho mỗi sequence:

- Gọi `D` là số quyết định routing hợp lệ trong phạm vi đang xét; `N` là số expert và `k = 2`.
- `f_m = N × count_selected(m) / (k × D)`, với count lấy từ top-k thực sự được chọn bằng `σ + β` và không truyền gradient qua count.
- `P_m = mean_decisions[σ_m / sum_all_experts(σ)]`; xác suất này không chứa `β`.
- `L_balance = α × sum_m(f_m × P_m)`.
- Shared pool: gộp các quyết định token×middle-step trong từng sequence rồi tính một loss cho pool.
- Per-layer MoE: tính riêng từng middle layer rồi lấy trung bình qua các layer.
- Cuối cùng lấy trung bình qua các sequence, chuẩn hóa nhất quán giữa microbatch/DDP ranks.
- “Sequence” là một hàng đầu vào: một fact ở E1 hoặc một hàng packed 512 token ở E2; bỏ PAD. Việc tính auxiliary loss không thay đổi forward routing hay document attention mask.

Cách cộng hoặc lấy trung bình qua depth làm thay đổi cường độ regularization, nên cần chốt trước khi so sánh các arm.

## 2. Dấu của phép kiểm tra Taylor T5 (§9)

Spec định nghĩa `c_m = -<g, k*u_m - y>`. Nếu `ΔL = L_counterfactual - L_normal` thì `c_m` dự đoán **mức giảm loss**, tức `-ΔL`.

§5 đã dùng `Spearman(c_m, -ΔL)`, nhưng T5 nói tương quan với “finite-difference loss changes” và kỳ vọng dương.

**Câu hỏi:** Chốt T5 dùng `Spearman(c_m, -ΔL)` giống §5, đúng không?

**Đề xuất:** dùng `-ΔL`; trong output lưu rõ cả `ΔL` và định nghĩa dấu để tránh đánh giá ngược chất lượng nhãn.

## 3. Loss và scale gradient để tạo Taylor labels (§2.3)

Spec nói scale của `g` không quan trọng vì score được chuẩn hóa. Điều này chỉ gần đúng khi epsilon không đáng kể: với mẫu số `std(c) + 1e-6`, thay đổi scale vẫn thay đổi phân phối nhãn nếu `std(c)` nhỏ.

Gradient lấy từ loss trung bình còn phụ thuộc số token/answer được chấm trong microbatch. Vì vậy cùng một ví dụ có thể nhận nhãn khác khi đổi cách chia batch, dù global batch không đổi.

**Câu hỏi:** Có đồng ý quy ước sau không?

- `g` dùng tạo nhãn là gradient của **tổng task NLL trên các target hợp lệ**, không bao gồm balance loss hoặc Taylor replay loss.
- Nếu backward dùng task loss đã chia trung bình/gradient accumulation, hoàn nguyên đúng hệ số đó khi lấy `g` để tạo nhãn. Không thay đổi gradient dùng cập nhật model.
- Chuẩn hóa score với population standard deviation (`unbiased=False`) và giữ epsilon `1e-6` như spec.
- Kiểm tra cùng dữ liệu/weights cho nhãn tương đương khi đổi cách chia microbatch; lưu thống kê `std(c)` để phát hiện nhãn gần uniform do epsilon.

Đây là đề xuất làm rõ định nghĩa nhãn, không phải thay đổi learning rate hay thêm một arm.

## 4. Đích và cơ chế backup checkpoint ra ngoài node (§0)

Spec yêu cầu backup final checkpoint của từng run, thêm pre-decay checkpoint của E2, cùng config/metrics/eval/routes và kiểm tra SHA-256.

Hai ràng buộc hiện có:

- [AGENT_GUIDE.md](AGENT_GUIDE.md), mục “No direct outbound internet connections from the GPU machine”, cấm GPU workload tự upload ra ngoài.
- [commands.md](commands.md) quy định kênh lấy kết quả nhỏ; không dùng việc chia file thành nhiều mảnh để biến một đợt tải checkpoint lớn thành tải nhỏ.

**Câu hỏi:** Dùng đích nào và cơ chế truyền dung lượng lớn nào đã được hỗ trợ?

- Dev machine: cần đường dẫn đích và kênh truyền/controller được phép sử dụng.
- Private Hugging Face: cần repo ID thuộc tài khoản chủ project và cơ chế upload được phép từ controller/dev machine; không ghi token vào tài liệu hoặc Git.

**Đề xuất:** chốt đường backup và thử truyền/xác minh một artifact trước Phase 0. Checkpoint pre-decay phải có đủ optimizer/scheduler/RNG và vị trí dữ liệu để resume. Mỗi lần copy xong ghi manifest, kích thước, vị trí và SHA-256 đã đối chiếu; bản chỉ nằm trong `_backup` trên cùng node chưa đáp ứng yêu cầu off-node.

## Cách hiểu mặc định, không cần mở thêm thí nghiệm

- E1 “answer position” trong phân tích routing là hidden state **dự đoán** answer: token `r` của atomic query hoặc `r2` của two-hop query. Không lấy state đã đọc token đáp án thật.
- Tính SHA trực tiếp từ spec hiện có để ghi vào config/log/report; nếu spec được sửa sau khi chốt câu hỏi thì dùng SHA mới và giữ lịch sử thay đổi.
- Epoch/token budget được chọn theo throughput đo ở Phase 0; chưa cam kết thời gian thực tế từ các con số ước lượng.
- Record-and-continue áp dụng cho các kết quả/check khoa học theo spec; vẫn giữ quy tắc ownership và xử lý lỗi hạ tầng của project.

**Còn lại:** chốt câu 1–3 trước phần implementation phụ thuộc các định nghĩa đó; chốt câu 4 trước khi chạy GPU. Chưa có thay đổi code train, lệnh runner hoặc workload remote.
