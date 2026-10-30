# Stage1b FineWeb-Edu — REPORT_interim

30/10/2026, 15:00 SGT. **Báo cáo interim một phần: Part1 training đã hoàn tất; downstream thất bại numerical gate trước scoring.** Part2 chưa chạy. Đây không phải chứng nhận hoàn tất Part1 downstream hoặc toàn Stage1b.

## Trạng thái downstream và điểm cần quyết định

DCLM P300 regression PASS: cả 9 primary task accuracies khớp Stage1; LL max difference = 0.0 trên 78.259 request; score-file SHA giống hệt. Tuy nhiên, gate thực tế của **FWE P300 parent** lúc 14:04 SGT không đạt: batch1-vs-batch2 và production-batch2 error = **0.011164188385009766 nats/token**, ngưỡng **0.01**. Manual-reference error = 0.0, greedy mismatches = 0, 12 request cố định, không truncation. Hai kiểm tra này dùng checkpoint khác nhau; regression PASS không thay thế FWE gate.

Job exit1 trước scoring; chưa có score bốn model, paired CI hoặc gate của student. Supervisor đã xác minh 8 GPU trống, khôi phục burn lúc14:05:36 SGT; kiểm tra độc lập14:31:32 SGT thấy guard released, workers723690–723697, cycles1780→1790, payload1976.43→1987.53 GiB. Source `05b0d0a944b4f01e947e0ea50fea62a01510da54`, 122 files, giữ nguyên 66 eval/model hashes của regression. [Bằng chứng](results/a2_stage1b_fwe_20261030/downstream_part1_gate_failure/verification.json).

Chưa sửa threshold, batch size, precision hay numerical core. Latest user exception cho phép tiếp tục Part2 khi **regression accuracy FAIL**; regression ở đây PASS, còn FWE numerical gate FAIL. Vì spec yêu cầu dừng và báo khi không thực hiện đúng protocol, cần quyết định riêng cho tình huống này trước khi tiếp tục GPU work. Watcher cũ vẫn giữ; không restart job lỗi.

## Dữ liệu, tests và parents

Heal stream: `HuggingFaceFW/fineweb-edu`, prepared revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Release gốc `v1.0.0`, revision `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`: **60 file được chọn có byte/LFS identity giống nhau, xác nhận heal stream thuộc release các parent được train trên**. `FineWeb-Edu-1.4T` mô tả budget, không phải revision. [Release proof](../resources/a2_stage1b_fwe_release_20261030.json).

Shuffle document một lần `numpy.random.default_rng(2030)`, pack4096, prefix lồng nhau; 29,999,992,832 tokens/7,324,217 sequences. Manifest `28362487768d9ceb285a06788f4d9589f1f692be8185d3091abc7ffc6ed50a3f`. Không thay đổi hay chuẩn bị lại corpus gốc.

| Check | Kết quả |
|---|---|
| T1 | P300/P1T/S72 PASS; precision gaps 0.0000913465 / 0.0001567571 / 0.0000776934 nats |
| T3 | PASS, không có failure reason; S72/X72 đủ điều kiện |
| T7 | PASS; document-ID overlap0, content-hash overlap0 |
| T18 | 7,324,217 sequences; EOS29,155,864, share0.000971862366; cả ba parent có FWE held-out loss thấp hơn CulturaX, không flag |
| T17 | Đã rerun trước science; delta−0.000144556073 trong±0.002 |
| Training | Full LR, exact8rank order, roundtrip0, không divergence ở cả10fits Part1 |

| Parent | CulturaX final | CulturaX tuning | FWE held-out | S1 pruned final |
|---|---:|---:|---:|---:|
| P300 | 2.860631326 | 2.841494222 | 2.308842449 | 3.822488626 |
| P1T | 2.808707334 | 2.790512734 | 2.256209723 | 4.124982157 |
| S72 | 3.042152002 | 3.024812803 | 2.489572923 | 4.007307971 |

Parent final NLLs P300/P1T khớp chính xác Phase0A. S72−P300 annealing gain=0.18152067608948386 nats. T14a identical-rerun noise−0.000006436, alternate-order−0.002285655; đây là hai phép đo đơn lẻ, không phải CI/variance nhiều seed.

## Training results và quyết định NLL

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

FWE held-out tại 0.2B: Δ_post(P300)=+0.196239975979; Δ_post(P1T)=+0.222592758859; E=+0.026352782881 (diagnostic).

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


## Đối chiếu và phạm vi suy luận

| Budget | FWE E | DCLM E | FWE−DCLM |
|---|---:|---:|---:|
| 0.2B | 0.026224222 | 0.021096293 | +0.005127928 |
| 0.8B | Chờ | 0.020786423 | Chờ |
| 3.2B | 0.023461150 | 0.021670885 | +0.001790266 |

FWE A_init ở mọi budget và shrink ratios chờ X72/Part2. DCLM A_init(.2/.8/3.2)=0.072724704/0.038910029/0.012113132; không thay số này cho FWE. Nhãn training-length FWE **persists → replicated** giữ nguyên; downstream thất bại không đổi nhãn NLL.

Điều kiện CulturaX lịch sử: Phase0A DCLM/S1 E(.05B)=0.019603985 và E(.2B)=0.010876774, LR3e-4; Stage1 DCLM E(.2B)=0.021096293, LR1e-3. Khác corpus healing và lựa chọn LR, không phải ablation riêng corpus. CulturaX final là tập quyết định; own-corpus held-out chỉ diagnostic. P1T đã đọc khoảng71% FWE so với21% của P300 (DCLM25%/7.5%); không diễn giải exposure như nguyên nhân đã được cô lập, không tự chạy exposure test.

## Provenance và compute

Part1 thực sự chạy **`7dd9da088ba5af581c295775819319ce54b2025c`**, trong khi trường provenance kế thừa preflight **`9f5601d97ddf2940b32ba5e17de8d21b7c7fe1c0`**; **27 core-file hashes trùng nhau**. Published results giữ nguyên. Part2 chưa chạy, commit thực tế chưa có. [Code audit](A2_STAGE1B_CODE_AUDIT_20261030.md), [runtime/regression audit](A2_STAGE1B_FOLLOWUP_REPLY_20261030.md).

Spec v1.1 SHA `b2121a4f2859f99db421273082c6be035dc5e833af6d1d071f5ed4cd381ed483`; v3.3 SHA `9ebe0d0a377f0107f7058bf5e4899b5380cbe5f4f518b09091ac533d2c9f254e`.

Training Part1 terminal12:43:17SGT; downstream failure14:04:31SGT, burn restored14:05:36SGT. Ledger bảo thủ lúc downstream launch38,275.421714sec (~10.63nodeh), gồm cả thời gian chờ; dự trữ timeout7,200sec+cleanup660 dưới20/24h. Ledger bảo thủ tới15:00 khoảng11.60nodeh. Không dùng ETA20–21SGT cũ làm cam kết sau failure. Trước Part2 phải tính lại budget, báo nếu projection vượt20h, tuyệt đối giữ hard24h và deadline31Oct04:06:27SGT.

[Decision JSON](decision_stage1_fwe_interim.json). Downstream scores/paired CI: **không có**, không điền zero hoặc suy ra từ DCLM. Full Stage1b chưa hoàn tất.
