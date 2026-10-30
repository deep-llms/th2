# Stage1b FineWeb-Edu: chọn LR và hai arm 3.2B

Cập nhật tài liệu: **30/10/2026, 10:56 SGT**. Snapshot đã xác minh gần nhất: **10:31:35 SGT**.

**Chưa đủ kết quả cuối của cả hai arm 3.2B.** Sáu run grid ban đầu và hai run mở rộng đã hoàn tất. P300 3.2B đã hoàn tất; P1T 3.2B đang ở bước **4.030/12.207 (33,0%)** trong snapshot trên. ETA P1T **12:45–13:00 SGT**, còn phụ thuộc validation/save. Các ô “Chờ” dưới đây chưa có kết quả, không phải giá trị bằng 0.

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
| Trạng thái | Hoàn tất, xác minh 09:31 SGT | Đang chạy, 4.030/12.207 bước lúc 10:31 SGT |
| Parent CulturaX final NLL | 2.860631326 | 2.808707334 |
| Student CulturaX tuning NLL | 2.936589511 | Chờ |
| Student CulturaX final NLL | 2.955812353 | Chờ |
| Δ_post CulturaX final | +0.095181027 | Chờ |
| Parent FWE held-out NLL | 2.308842449 | 2.256209723 |
| Student FWE held-out NLL | 2.406616605 | Chờ |
| Δ_post FWE held-out | +0.097774156 | Chờ |

**E tại 3.2B và nhãn kết luận training-length chưa xác định**, vì thiếu kết quả cuối P1T. Downstream bốn model cũng chưa chạy; tài liệu này chưa phải `REPORT_interim.md` hoàn chỉnh theo spec.

## Kiểm chứng và nguồn

Cả tám run grid và P300 3.2B đã được kiểm tra đủ lịch LR, thứ tự dữ liệu chính xác trên 8 rank, sai khác save/reload bằng 0, không divergence. Snapshot P1T hiện tại có 8 GPU hoạt động 98–100%; source, inputs, runtime, ownership và chín receipt đã hoàn tất không đổi.

- [Receipt 10:31 SGT](results/a2_stage1b_fwe_20261030/part1_1031/receipt.json), [SHA-256 của receipt](results/a2_stage1b_fwe_20261030/part1_1031/receipt.sha256).
- [Grid note và phép tính E](results/a2_stage1b_fwe_20261030/part1_0631/grid_note.json).
- [Preflight và đo parent](A2_STAGE1B_PREFLIGHT_20261030.md).
- Training source: `7dd9da088ba5af581c295775819319ce54b2025c` (124 file đã xác minh).
- Raw health SHA-256: `1455afb0f014ff231dc43396eaf6548a0fe845b500d5b30faed81f26b02a0d36`.

Số hiển thị được làm tròn 9 chữ số thập phân; receipt lưu độ chính xác đầy đủ. Khi P1T hoàn tất, cập nhật bảng 3.2B và E từ receipt cuối đã kiểm chứng.
