# Stage1b FineWeb-Edu: chọn LR và ba arm 3.2B

Cập nhật: **30/10/2026, 21:01:08 SGT**, từ healtha10 đã xác minh. Part1 training hoàn tất lúc **12:43:17 SGT**; X72 3.2B đã hoàn tất và P300 0.8B đang chạy.

**Part1 đã đủ sáu run grid ban đầu, hai run mở rộng và P300/P1T 3.2B.** Đủ lịch LR, exact 8-rank order, save/reload sai khác 0 và terminal exit 0 của Part1 đã kiểm tra. Native burn handoff của Part1 được xác minh độc lập. X72 3.2B cũng đã được kiểm tra đầy đủ; tổng 11/15 fit đã xác minh. Bốn fit ngắn Part2, downstream và báo cáo toàn Stage1b còn chờ.

## Thiết lập và cách đọc

Theo [Stage1b spec v1.1](A2_stage1b_fwe_spec.md): heal trên FineWeb-Edu, layer set S1, shuffle một lần ở cấp document bằng seed 2030, dùng các prefix lồng nhau. Mỗi run grid có ngân sách 0.2B token; hai run chính có ngân sách 3.2B token mỗi run.

NLL càng thấp càng tốt. **Chọn LR bằng trung bình CulturaX tuning NLL của P300 và P1T**. CulturaX final dùng để tính kết quả chính; FineWeb-Edu held-out chỉ là chỉ số chẩn đoán. `Δ_post = NLL_student − NLL_parent`; `E = Δ_post(P1T) − Δ_post(P300)`. E dương nghĩa là P1T còn chịu mức tăng NLL lớn hơn P300 so với parent tương ứng.

## Sáu run chọn LR ban đầu

| Arm | LR | CulturaX tuning NLL | CulturaX final NLL | FWE held-out NLL | Δ_post (CulturaX final) |
|---|---:|---:|---:|---:|---:|
| P300 | 0.0001 | 3.151107981 | 3.169482734 | 2.624746937 | +0.308851408 |
| P1T | 0.0001 | 3.125343523 | 3.142402422 | 2.597472721 | +0.333695089 |
| P300 | 0.0003 | 3.084298514 | 3.103321913 | 2.553008765 | +0.242690587 |
| P1T | 0.0003 | 3.053515231 | 3.071785913 | 2.521224369 | +0.263078580 |
| P300 | 0.001 | 3.037060038 | 3.056353314 | 2.505082425 | +0.195721988 |
| P1T | 0.001 | 3.011759344 | 3.030653544 | 2.478802482 | +0.221946210 |

## Hai run mở rộng theo quy tắc biên

LR tốt nhất của grid ban đầu nằm ở biên trên 1e-3, nên chạy thêm đúng một cặp ở 3e-3 theo spec.

| Arm | LR | CulturaX tuning NLL | CulturaX final NLL | FWE held-out NLL | Δ_post (CulturaX final) |
|---|---:|---:|---:|---:|---:|
| P300 | 3e-3 | 3.052761722 | 3.071304613 | 2.520658197 | +0.210673287 |
| P1T | 3e-3 | 3.051705074 | 3.069546812 | 2.518718774 | +0.260839479 |

| LR | Trung bình tuning NLL của hai arm |
|---|---:|
| 0.0001 | 3.138225752 |
| 0.0003 | 3.068906872 |
| 0.001 | 3.024409691 |
| 0.003 | 3.052233398 |

**LR chung được chọn: 1e-3. LR tốt nhất riêng của cả P300 và P1T cũng là 1e-3.** Cặp mở rộng không cải thiện kết quả; không mở rộng thêm. Lựa chọn này trùng với DCLM nhưng được tính lại độc lập trên FineWeb-Edu.

Tại 0.2B và LR đã chọn:

- CulturaX final: Δ_post(P300) = **+0.195721988**; Δ_post(P1T) = **+0.221946210**; **E = +0.026224222**.
- FWE held-out: **E = +0.026352783** (chẩn đoán).

## Hai arm 3.2B, LR 1e-3

| Chỉ số | P300 | P1T |
|---|---:|---:|
| Trạng thái | Hoàn tất, xác minh 09:31 SGT | Hoàn tất, terminal xác minh 13:00 SGT |
| Parent CulturaX final NLL | 2.860631326 | 2.808707334 |
| Student CulturaX tuning NLL | 2.936589511 | 2.908373912 |
| Student CulturaX final NLL | 2.955812353 | 2.927349511 |
| Δ_post CulturaX final | +0.095181027 | +0.118642177 |
| Parent FWE held-out NLL | 2.308842449 | 2.256209723 |
| Student FWE held-out NLL | 2.406616605 | 2.376905284 |
| Δ_post FWE held-out | +0.097774156 | +0.120695562 |

**E tại 3.2B = +0.023461150 nats**, nhãn training-length **persists**, đối chiếu DCLM là **replicated** theo spec. FWE held-out E = **+0.022921405** (chẩn đoán). DCLM E3.2 = +0.021670885; FWE − DCLM = **+0.001790266**. A_init(3.2B) đã có bên dưới; xu hướng 0.8B và toàn bộ Stage1b còn chờ.

DCLM P300 regression đã PASS: cả 9 task accuracy khớp chính xác, LL max difference bằng 0. FWE downstream ban đầu thất bại tại numerical gate trước scoring; theo quyết định mới, sẽ chạy một lượt sáu model sau Part2, với kiểm tra đầy đủ P300 batch1/batch2. Chi tiết và receipt lỗi ở [REPORT.md](REPORT.md); chưa có FWE downstream scores/CI.

## X72 3.2B và A_init

X72 đã hoàn tất 12.207 updates / 3.199.991.808 tokens tại LR 1e-3, không divergence; full LR schedule, exact 8-rank order và save/reload sai khác 0 đã xác minh.

| Chỉ số | X72 |
|---|---:|
| CulturaX tuning NLL | 2.949474459 |
| CulturaX final NLL | 2.968806004 |
| Δ_post so với S72 parent | −0.073345998 |
| FWE held-out NLL | 2.420236999 |
| Thời gian training (giây) | 11721.795334 |

**A_init(3.2B) = NLL(X72 student) − NLL(P300 student) = +0.01299365118963669 nats**, nhãn **“anneal first”** theo ngưỡng đã đăng ký. X72 có NLL cao hơn P300 student; đây là đối chiếu recipe đã đăng ký, không tách riêng tác động của LR state. FWE held-out A_init = **+0.013620394355744914**, chỉ dùng chẩn đoán. DCLM A_init(3.2B) ≈ +0.012113132, cùng nhãn; A_init ở 0.2B/0.8B và tỷ lệ thu hẹp của FWE chưa có.

[Receipt healtha10](results/a2_stage1b_fwe_20261030/part2_2101/receipt.json), [X72 và A_init đầy đủ](results/a2_stage1b_fwe_20261030/part2_2101/x72_3_2b_summary.json), [SHA-256](results/a2_stage1b_fwe_20261030/part2_2101/SHA256SUMS). Part2 còn chạy P300 0.8B (1.480/3.051 bước tại snapshot), rồi P1T 0.8B, X72 0.8B, X72 0.2B. Chưa có terminal/burn handoff của Part2.

## Kiểm chứng và nguồn

Cả tám run grid và hai run 3.2B đã qua kiểm tra: full LR schedule, exact 8-rank order, roundtrip 0, không divergence. Native terminal exit 0; guard đã released; 8 burn worker đã xác minh, collective cycles tăng 1140→1150 và payload 1265.80→1276.90 GiB.

- [Terminal/handoff verification](results/a2_stage1b_fwe_20261030/part1_final/verification.json), [fit summary](results/a2_stage1b_fwe_20261030/part1_final/fit_summary.json), [SHA-256](results/a2_stage1b_fwe_20261030/part1_final/SHA256SUMS).
- [Training decision gốc](results/a2_stage1b_fwe_20261030/part1_final/decision_stage1_fwe.json).
- [Grid note](results/a2_stage1b_fwe_20261030/part1_0631/grid_note.json), [preflight/parents](A2_STAGE1B_PREFLIGHT_20261030.md).
- Actual frozen training source: `7dd9da088ba5af581c295775819319ce54b2025c` (124 files). Result provenance giữ preflight `9f5601d97ddf2940b32ba5e17de8d21b7c7fe1c0` sau khi so khớp **27 core-file hashes**. Published results không sửa. Part2 thực thi frozen commit `42f7820ce72d932ff3d75edc96e16538dbd85df5` (126 files), vẫn giữ provenance preflight và 27 core-file hashes khớp.
- Raw terminal health SHA: `7247425579c0a24acd6c742466b4a90dea9499347964dc2f8abb27238247018d`.

Số hiển thị làm tròn 9 chữ số thập phân; receipt lưu độ chính xác đầy đủ. [Audit code và T17](A2_STAGE1B_CODE_AUDIT_20261030.md).
