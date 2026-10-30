# Stage1b FineWeb-Edu: chọn LR và hai arm 3.2B

Cập nhật: **30/10/2026, 13:00:54 SGT**, từ receipt terminal đã xác minh. Part1 training hoàn tất lúc **12:43:17 SGT**.

**Đã đủ kết quả của sáu run grid ban đầu, hai run mở rộng và cả hai arm 3.2B.** Đủ lịch LR, exact 8-rank order, save/reload sai khác 0 và terminal exit 0 đã kiểm tra. Native burn handoff được xác minh độc lập. Downstream, Part2 và báo cáo toàn Stage1b còn chờ.

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

**E tại 3.2B = +0.023461150 nats**, nhãn training-length **persists**, đối chiếu DCLM là **replicated** theo spec. FWE held-out E = **+0.022921405** (chẩn đoán). DCLM E3.2 = +0.021670885; FWE − DCLM = **+0.001790266**. A_init và xu hướng 0.8B còn chờ Part2; chưa kết luận toàn bộ Stage1b.

Re-score DCLM P300 parent phải hoàn tất trước FWE downstream theo [follow-up](A2_STAGE1B_FOLLOWUP_REPLY_20261030.md). Downstream bốn model chưa chạy, nên tài liệu này chưa thay thế `REPORT_interim.md`.

## Kiểm chứng và nguồn

Cả tám run grid và hai run 3.2B đã qua kiểm tra: full LR schedule, exact 8-rank order, roundtrip 0, không divergence. Native terminal exit 0; guard đã released; 8 burn worker đã xác minh, collective cycles tăng 1140→1150 và payload 1265.80→1276.90 GiB.

- [Terminal/handoff verification](results/a2_stage1b_fwe_20261030/part1_final/verification.json), [fit summary](results/a2_stage1b_fwe_20261030/part1_final/fit_summary.json), [SHA-256](results/a2_stage1b_fwe_20261030/part1_final/SHA256SUMS).
- [Training decision gốc](results/a2_stage1b_fwe_20261030/part1_final/decision_stage1_fwe.json).
- [Grid note](results/a2_stage1b_fwe_20261030/part1_0631/grid_note.json), [preflight/parents](A2_STAGE1B_PREFLIGHT_20261030.md).
- Actual frozen training source: `7dd9da088ba5af581c295775819319ce54b2025c` (124 files). Result provenance giữ preflight `9f5601d97ddf2940b32ba5e17de8d21b7c7fe1c0` sau khi so khớp **27 core-file hashes**. Published results không sửa. Commit thực tế Part2 sẽ ghi khi launch.
- Raw terminal health SHA: `7247425579c0a24acd6c742466b4a90dea9499347964dc2f8abb27238247018d`.

Số hiển thị làm tròn 9 chữ số thập phân; receipt lưu độ chính xác đầy đủ. [Audit code và T17](A2_STAGE1B_CODE_AUDIT_20261030.md).
