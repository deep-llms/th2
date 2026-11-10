# A2 — Family v1.7 và Depth v1.1

Cập nhật: **10/11/2026 15:29 SGT**.

**Trạng thái:** Family F0–F8 (các nhánh có trigger theo spec) và toàn bộ Depth A/B/C + downstream đã hoàn tất. Backup khoa học đã tải về, đối chiếu SHA và replay độc lập. F5 không chạy vì trigger không bật; đây là kết quả đúng kế hoạch, không phải phần còn thiếu.

Báo cáo này giữ nguyên các kết quả và flags F0–F6. F7/F8 là phần mở rộng được chủ dự án yêu cầu, không thay thế các kết luận đăng ký trước. “Hoàn tất/replay đạt” nói về thực thi và khả năng tái lập số liệu; không xóa các giới hạn khoa học ghi bên dưới.

## 1. Kết quả chính

| Phần | Kết quả | Diễn giải |
|---|---|---|
| Family SmolLM3, FWE | E(3,2B) = 0,009719015; FE1 `persists`, kèm `not converged` | E giảm theo budget nhưng chưa bị xóa; các flag FT2 vẫn giữ. |
| Family mixed web | FE4 `robust`; chênh E với FWE ở 0,2B = +0,001492853 | Nhỏ hơn ngưỡng 0,003; F5 0,8B không được kích hoạt. Các giới hạn mixture vẫn giữ. |
| Family F6 | DCLM 0,2→0,8B `flat`; FWE `falls`; joint reading `mixed` | Kết luận gốc không đổi. |
| Family F7 | DCLM stable E(3,2B) = 0.005171953; `falls` | Kết luận phụ so với 0,8B; không đổi FE6 gốc/joint reading. |
| Family F8 | DCLM four-cut 3,2B: `mixed` | Không tăng đơn điệu qua L3/L6/L9/L12; không thay thế kết luận Why ở 0,8B. |
| Depth A | A1–A4 đều false | Các điều kiện tổng hợp đăng ký trước không thỏa mãn; không phải lỗi chạy. |
| Depth B | B1 true, B2 false; E `persists` ở cả hai block và dataset | Đã hoàn tất 12 fits 0,2B. |
| Depth C | E = 0.020443866 (`persists`); A_init = 0.001229187 (`no effect`) | C1/C2 đều true; cả ba arm đã rerun đủ 12,8B trên wc69. |

NLL và E/A_init có đơn vị nats trên prediction token. PPL = exp(NLL); NLL/PPL càng thấp càng tốt. E = Δ_post(long) − Δ_post(short), với Δ_post tính từ parent riêng của từng arm. A_init = NLL(X72 student) − NLL(P300 student).

## 2. Family v1.7 — các phần bổ sung F8 và F7

F8 chạy trước F7. Sáu fits đều đủ 12.207 updates (3.199.991.808 token), LR 0,001, seed 0, stream seed 2030. FT10 được bỏ qua theo spec vì vẫn cùng node wc69.

| Run | Parent NLL | Student NLL | Δ_pre | Δ_post |
|---|---:|---:|---:|---:|
| ST1-dclm-S188-S1-3.2B-lr1e-3 | 2.886706698 | 2.941622756 | +1.125950111 | +0.054916058 |
| ST1-dclm-S72-S1-3.2B-lr1e-3 | 2.908575296 | 2.958319402 | +0.945968926 | +0.049744105 |
| Y3-dclm-P1T-L3-3.2B | 2.686908992 | 2.755552615 | +0.323383519 | +0.068643623 |
| Y3-dclm-P1T-L9-3.2B | 2.686908992 | 2.871552250 | +3.249153015 | +0.184643258 |
| Y3-dclm-P300-L3-3.2B | 2.737793032 | 2.789057126 | +0.273158845 | +0.051264094 |
| Y3-dclm-P300-L9-3.2B | 2.737793032 | 2.897751017 | +2.581002209 | +0.159957985 |

| Cặp ở 3,2B | E | CI95 E | E/G | Phần parent gain giữ lại |
|---|---:|---|---:|---:|
| F8-L3 | 0.017379529 | [0.016971671, 0.017802501] | 0.341552 | 0.658448 |
| F8-L9 | 0.024685274 | [0.024083793, 0.025277339] | 0.485128 | 0.514872 |
| F7 stable S72→S188 | 0.005171953 | [0.004717248, 0.005627455] | 0.236501 | 0.763499 |

| Cut DCLM, 3,2B | E | Phần parent gain giữ lại | Nguồn |
|---|---:|---:|---|
| L6 | 0.021670885 | 0.574112 | Stage 1, node 6y3o |
| L12 | 0.023323835 | 0.541628 | Why L12, wc69 |
| L3 | 0.017379529 | 0.658448 | Family F8, wc69 |
| L9 | 0.024685274 | 0.514872 | Family F8, wc69 |

Slope E theo số lớp bỏ: 0.000694910254 nats/lớp; E(L12)−E(L3) = 0.005944306. L3 và L9 đều `flat` khi so 0,8B→3,2B theo ngưỡng 0,003.


| Stable pair | 0,2B E | 0,8B E | 3,2B E |
|---|---:|---:|---:|
| dclm | 0.011131974 | 0.008627345 | 0.005171953 |
| fwe | 0.014780744 | 0.008610999 | 0.005028513 |

F7 DCLM: E(3,2B)−E(0,8B) = -0.003455392; E/Δ_post(S72) = 0.103971. F6 báo cáo cả full/reduced FWE; audit kết luận overlap immaterial. Các trị số, full/reduced và flags gốc nằm nguyên vẹn trong phụ lục Family.

## 3. Depth A và B

Part A: đủ 144 cấu hình pruned và 12 parent, 156 mảng; replay độc lập 1.938 so sánh PASS. DT1–DT3 và các điều kiện dự đoán được giữ theo báo cáo gốc. Part B: 12 fits + DT4, 762 updates/fit; 205 file/39 mảng được tải và replay 504 kiểm tra PASS. DT4 sai khác +0,000394060162 so reference, dưới ngưỡng 0,002; đây là một cặp kiểm tra, không phải cận sai số tổng quát.

Bảng độ sâu, contrast tương đối, correlations và từng clause A1–A4; bảng B cạnh S1 được giữ trong phụ lục A/B của file này.

## 4. Depth C — bộ rerun 12,8B trên wc69

P300/P1T là tên checkpoint parent đã pretrained 300B/1T. Các lượt dưới đây là **healing lại 12,8B**, không phải pretrain lại từ đầu 300B/1T. X72 khởi tạo từ S72 và học từ teacher P300; hai arm còn lại dùng parent tương ứng làm khởi tạo và teacher.

Đủ 48.828 updates/arm, global batch 64 × 4.096, LR 0,001, warmup 977 rồi linear decay về 0; seed/data_seed = 0. DT6 xác nhận nested stream với 3.124.992 sequences và prefix 781.248 sequences đúng hash lịch sử. Các lượt tạm dừng để ưu tiên Why/Family được resume đúng lịch LR và data order.

| Arm | Parent NLL | Final NLL | PPL | Δ_post |
|---|---:|---:|---:|---:|
| P300 | 2.737793032 | 2.816504216 | 16.718305 | +0.078711184 |
| P1T | 2.686908992 | 2.786064041 | 16.217064 | +0.099155050 |
| X72 | 2.908575296 | 2.817733403 | 16.738867 | -0.090841894 |

| Heal budget, DCLM S1 | E | A_init | Nguồn |
|---|---:|---:|---|
| 0.2B | 0.021096293 | 0.072724704 | Stage 1 lịch sử |
| 0.8B | 0.020786423 | 0.038910029 | Stage 1 lịch sử |
| 3.2B | 0.021670885 | 0.012113132 | Stage 1 lịch sử |
| 12.8B | 0.020443866 | 0.001229187 | Bộ rerun đủ trên wc69 |

E(12,8B)/E(3,2B) = 0.943379; A_init(12,8B)/A_init(3,2B) = 0.101476.

- E: **+0.020444 [+0.019983, +0.020923]**, nhãn **persists**; C1 (`E ≥ 0,008`) = **true**.
- A_init: **+0.001229 [+0.001048, +0.001415]**, nhãn **no effect**; C2 (`A_init < 0,01`) = **true**.

CI95 trên đây được tính độc lập bằng paired packed-sequence bootstrap 10.000 lần, seed 0, 2.461 rows/10.077.795 target tokens. Mỗi row có 4.095 targets nên mean theo row bằng mean token-weighted. CI mô tả nhiễu evaluation, không phải sai số giữa seed training hay mẫu document độc lập.

P300 resume từ các checkpoint 24.000 và 45.000; P1T resume từ 15.000. Node kết thúc toàn bộ Depth/downstream lúc **10/11/2026 07:18:31 SGT**; native burn handoff xác minh lúc **07:19:42 SGT**, được đối chiếu lại ownership lúc **15:05 SGT**.

## 5. Depth downstream — batch 2, zero-shot

Tám bài multiple-choice và LAMBADA, cùng bộ prompt/request cho ba parents và ba students; 26.618 ví dụ/model. MC mean là trung bình đều tám accuracy, không gộp tất cả câu thành một tỷ lệ. Các giá trị bảng dưới là %.

| Model | MC mean | LAMBADA accuracy |
|---|---:|---:|
| P300-parent | 62.4059 | 65.4764 |
| P1T-parent | 64.4467 | 67.6499 |
| S72-parent | 59.1902 | 57.0541 |
| P300-student | 59.4074 | 62.0415 |
| P1T-student | 61.5976 | 64.8554 |
| X72-student | 60.0587 | 61.9057 |

**Paired contrasts**, đơn vị điểm phần trăm; số trong ngoặc là CI95. “Drop” = parent − student, dương nghĩa là accuracy giảm. X72−P300 accuracy là hiệu trực tiếp giữa hai students, dương nghĩa X72 tốt hơn.

| Contrast | MC mean: estimate [CI95] | LAMBADA: estimate [CI95] |
|---|---|---|
| P300_drop | +2.998552 [+2.422556, +3.576034] | +3.434892 [+2.542208, +4.327576] |
| P1T_drop | +2.849032 [+2.265243, +3.446791] | +2.794489 [+1.901805, +3.725985] |
| X72_drop_from_S72 | -0.868538 [-1.518559, -0.192070] | -4.851543 [-5.957695, -3.784203] |
| P1T_minus_P300_drop | -0.149520 [-0.937097, +0.634199] | -0.640404 [-1.843586, +0.563264] |
| X72_minus_P300_accuracy | +0.651349 [+0.260977, +1.039211] | -0.135843 [-0.776247, +0.504560] |
| X72_minus_P300_drop | -3.867090 [-4.633348, -3.098690] | -8.286435 [-9.528430, -7.025034] |

P1T−P300 drop có CI chứa 0 trên MC mean; không diễn giải NLL E thành một hiệu ứng downstream chắc chắn. X72 có MC mean cao hơn P300 student trong screen này; đối chiếu LAMBADA riêng theo bảng. Đây là phần reported-only theo spec. Paired bootstrap 10.000 lần, seed 0, giữ pairing trong mỗi task; CI chỉ phản ánh benchmark items.

### Accuracy từng task (%)

| Task | P300 parent | P1T parent | S72 parent | P300 student | P1T student | X72 student |
|---|---:|---:|---:|---:|---:|---:|
| hellaswag | 65.0568 | 68.5919 | 58.0761 | 60.1773 | 62.7863 | 60.2071 |
| arc_easy | 67.0034 | 68.4343 | 61.2374 | 63.5522 | 65.1936 | 64.6044 |
| arc_challenge | 37.1160 | 40.3584 | 34.1297 | 32.5939 | 34.8976 | 34.3857 |
| piqa | 75.6801 | 76.5506 | 72.4157 | 73.0686 | 74.3199 | 73.7214 |
| winogrande | 62.0363 | 63.5359 | 59.2739 | 60.6946 | 63.4570 | 61.4049 |
| boolq | 62.3547 | 65.9021 | 63.0887 | 56.9725 | 61.9266 | 58.7462 |
| openbookqa | 39.2000 | 39.2000 | 36.8000 | 37.6000 | 37.4000 | 37.2000 |
| sciq | 90.8000 | 93.0000 | 88.5000 | 90.6000 | 92.8000 | 90.2000 |
| lambada_openai | 65.4764 | 67.6499 | 57.0541 | 62.0415 | 64.8554 | 61.9057 |

## 6. Kiểm chứng, backup và giới hạn

- Family v1.7: 152 file/48 raw rank traces khớp SHA; 576 worker checks, 247 replay checks theo run và 38 kiểm tra paired/readout PASS. Mọi giá trị F0–F6 và prefix báo cáo gốc được so sánh, giữ nguyên.
- Depth: 153 file trong 12 phần archive (290.407.890 byte) khớp SHA; replay Part C 133 checks PASS, sai khác tối đa 3,67×10⁻¹⁴. Downstream 303 checks PASS: likelihood→response coverage, accuracy từng câu, aggregate, pairing, bootstrap và so với fresh replay. Ba score identities khớp result/recipe và manifest final checkpoint tương ứng.
- Fresh node evidence: source Depth 191 + continuation 191 + Family 244 file không đổi; ownership của tám burn workers/launcher đúng native receipt; downstream đủ sáu model và không có flag mới. Pin checkpoint pause giữ metadata/JSON hashes; checkpoint gốc 15.000 đã được native rotation sau resume.
- Backup này lưu science/arrays/scores/request traces/config/provenance, **không chứa model weights hay optimizer nhiều GB**. Score/checkpoint SHA được đối chiếu manifest đã niêm phong; không tuyên bố đã tải hoặc load lại weights local.
- Các giới hạn Family FT2 (masking/reset fallback), FT8 tỷ lệ mixture prefix, uniform sampling trong shard pool thay vì toàn corpus, LR được chọn ở biên trên grid, và `not converged` của FE1 vẫn giữ. Các nhãn `unverified` kế thừa trong phụ lục không được xóa chỉ vì replay số liệu đạt.
- Khác node lịch sử: L6 3,2B từ 6y3o, các điểm F8 mới từ wc69; E là hiệu trong từng node. FWE overlap full/reduced và mọi qualification gốc được giữ trong phụ lục.
- Depth giữ deviation Breadth-before-Depth và thứ tự rerun X72 trước theo yêu cầu khôi phục của chủ dự án, các lần pause ưu tiên Why/Family, lỗi launcher ban đầu đã sửa, và lịch sử mất node/infra. Các scalar C cũ trên 6y3o không được ghép với bộ rerun wc69 để tính kết luận cuối.
- Đã có resume thực tế trên wc69; điều này không thay cho một controlled 8-GPU uninterrupted-versus-resumed bitwise-equivalence experiment. Không xóa giới hạn của các bằng chứng CPU/Gloo trước đó.

Ledger là report-only: Part A khoảng 1,0513 node-hour, DT4+B khoảng 3,1242 node-hour. Session Depth sau Family v1.7 kéo dài khoảng 10,17 giờ, gồm phần P1T còn lại và downstream; **không phải tổng compute của tất cả lượt Depth**. Các attempt/ledger lịch sử không được cộng thành tổng chính xác vì các session bị pause không đều ghi đủ elapsed. Ngân sách 42 giờ trong spec là ước lượng, không phải deadline.

### Provenance

- Family v1.7 spec SHA256: `70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209`.
- Depth v1.1 spec SHA256: `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`.
- Family archive SHA256: `688f2c4404f36102f7aa8249cceff101a65e4550d1e406a4456bae4045e6206c`.
- Depth archive SHA256: `4068966b2f98991017940ab2fc32b37457afd523aedbc342d7979f154ce5a9f7`.

Không còn lượt model hay replay bắt buộc đang chờ trong hai spec này. Các limitations/flags khoa học nói trên vẫn mở như qualification của kết quả.

---

## Phụ lục A — Báo cáo Family đầy đủ F0–F8

Nội dung native dưới đây giữ nguyên toàn bộ số, flags, per-run tables và readout gốc. Các dòng “remaining” hoặc thời điểm trong nội dung gốc là snapshot lúc viết; trạng thái kiểm chứng cuối ở mục 6 phía trên có ưu tiên.

# A2 Family v1.3

Spec SHA256: 40089a98702a285d38799adcdfa89c72c7eec2e75e627cfef8552448f9a593c0

Final readout: True; coverage: True.

- FE1: {'value': 0.00971901488669057, 'label': 'persists', 'budget': '3.2B', 'not_converged': True, 'unverified': True}
- FE2: {'label': 'changing', 'direction': 'down', 'difference': -0.006535701030237153, 'unverified': True}
- FE3: {'value': 0.007312437315304306, 'label': 'grows', 'budget': '3.2B', 'not_converged': False, 'unverified': True}
- data_order_noise: {'value': 0.0004369966630324712, 'flag': False, 'unverified': True}
- FE4: {'FE4': 'robust', 'difference': 0.0014928528840014443, 'direction': 'up', 'data_order_difference': 0.0004369966630324712, 'run_F5': False, 'unverified': True}

| Run | Parent NLL | NLL | Delta pre | Delta post | Exposure | Flags |
|---|---:|---:|---:|---:|---|---:|
| F2-fwe-k20-BS1-0.2B-lr0.0001-s2030 | 2.678332086 | 2.936645220 | 1.127589809 | 0.258313134 | not seen | 72 |
| F2-fwe-k20-BS1-0.2B-lr0.0001-s2031 | 2.678332086 | 2.935072186 | 1.127589809 | 0.256740100 | not seen | 72 |
| F2-fwe-k20-BS1-0.2B-lr2.5e-05-s2030 | 2.678332086 | 3.022093862 | 1.127589809 | 0.343761775 | not seen | 72 |
| F2-fwe-k20-BS1-0.2B-lr5e-05-s2030 | 2.678332086 | 2.975353894 | 1.127589809 | 0.297021808 | not seen | 72 |
| F2-fwe-k6-BS1-0.2B-lr0.0001-s2030 | 2.723293590 | 2.961746300 | 1.026185746 | 0.238452710 | not seen | 72 |
| F2-fwe-k6-BS1-0.2B-lr0.0001-s2031 | 2.723293590 | 2.959736269 | 1.026185746 | 0.236442679 | not seen | 72 |
| F2-fwe-k6-BS1-0.2B-lr2.5e-05-s2030 | 2.723293590 | 3.039208359 | 1.026185746 | 0.315914768 | not seen | 72 |
| F2-fwe-k6-BS1-0.2B-lr5e-05-s2030 | 2.723293590 | 2.996886943 | 1.026185746 | 0.273593353 | not seen | 72 |
| F2c-mix-k20-BS1-0.2B-lr0.0001-s2030 | 2.678332086 | 2.942843111 | 1.127589809 | 0.264511025 | not seen | 88 |
| F2c-mix-k6-BS1-0.2B-lr0.0001-s2030 | 2.723293590 | 2.966451338 | 1.026185746 | 0.243157747 | not seen | 88 |
| F3-fwe-k20-BS1-0.8B-lr0.0001-s2030 | 2.678332086 | 2.855236285 | 1.127589809 | 0.176904198 | not seen | 72 |
| F3-fwe-k20-BS1-3.2B-lr0.0001-s2030 | 2.678332086 | 2.801630386 | 1.127589809 | 0.123298300 | seen | 72 |
| F3-fwe-k6-BS1-0.8B-lr0.0001-s2030 | 2.723293590 | 2.886873065 | 1.026185746 | 0.163579475 | not seen | 72 |
| F3-fwe-k6-BS1-3.2B-lr0.0001-s2030 | 2.723293590 | 2.836872875 | 1.026185746 | 0.113579285 | seen | 72 |
| F4-fwe-k64-BS1-0.2B-lr0.0001-s2030 | 2.650000282 | 2.922642446 | 2.040374105 | 0.272642165 | not seen | 72 |
| F4-fwe-k64-BS1-0.8B-lr0.0001-s2030 | 2.650000282 | 2.837047271 | 2.040374105 | 0.187046990 | not seen | 72 |
| F4-fwe-k64-BS1-3.2B-lr0.0001-s2030 | 2.650000282 | 2.780611019 | 2.040374105 | 0.130610737 | seen | 72 |

| Contrast / data / budget / LR / order | Full | Reduced | Reduced minus full | Primary | 95% CI | E_pre | G | E/G | E/G 95% CI | Dropped docs / targets |
|---|---:|---:|---:|---|---|---:|---:|---:|---|---|
| E / fwe / 0.2B / 0.0001 / 2030 | 0.019860424 | 0.019863951 | +0.000003527 | full | [0.01911916670735676, 0.020606882993532052] | 0.101404063 | 0.044961504 | 0.44172064250308585 | [0.4288592363137444, 0.45492415253944585] | 3 / 1140 |
| E64 / fwe / 0.2B / 0.0001 / 2030 | 0.034189455 | 0.034189153 | -0.000000302 | full | [0.03328134518666394, 0.03519982542594845] | 1.014188359 | 0.073293309 | 0.4664744391201238 | [0.456911690728306, 0.4765215361054859] | 3 / 1140 |
| D64 / fwe / 0.2B / 0.0001 / 2030 | 0.014329031 | 0.014325201 | -0.000003829 | full | [0.013618781326432211, 0.015077983702223898] | 0.912784296 | 0.028331805 | 0.505757779707981 | [0.485923114448883, 0.5259532285139887] | 3 / 1140 |
| E / fwe / 0.2B / 0.0001 / 2031 | 0.020297421 | 0.020299540 | +0.000002119 | full | [0.019544622656445042, 0.02107822744457871] | 0.101404063 | 0.044961504 | 0.45143999403347645 | [0.4381819953742655, 0.4649124491244255] | 3 / 1140 |
| E / fwe / 0.2B / 2.5e-05 / 2030 | 0.027847007 | 0.027851073 | +0.000004066 | full | [0.027060587701655585, 0.028637616232778943] | 0.101404063 | 0.044961504 | 0.6193522089503595 | [0.605913142800819, 0.632707928341694] | 3 / 1140 |
| E / fwe / 0.2B / 5e-05 / 2030 | 0.023428455 | 0.023432508 | +0.000004053 | full | [0.022665189591775006, 0.02418887673043295] | 0.101404063 | 0.044961504 | 0.5210780960320273 | [0.5080179338980361, 0.5340904212884937] | 3 / 1140 |
| E / mix / 0.2B / 0.0001 / 2030 | 0.021353277 | 0.021356850 | +0.000003573 | full | [0.0206018207902273, 0.022108638345715054] | 0.101404063 | 0.044961504 | 0.4749235549969264 | [0.46206429704812074, 0.48769120473265326] | 3 / 1140 |
| E / fwe / 0.8B / 0.0001 / 2030 | 0.013324723 | 0.013327384 | +0.000002661 | full | [0.01259927161685909, 0.014057804107447808] | 0.101404063 | 0.044961504 | 0.29635848946144827 | [0.2828428722270998, 0.3101328950111438] | 3 / 1140 |
| E64 / fwe / 0.8B / 0.0001 / 2030 | 0.023467515 | 0.023467399 | -0.000000116 | full | [0.02261695158859439, 0.02442956901905133] | 1.014188359 | 0.073293309 | 0.32018631974025186 | [0.31069630253905317, 0.33040604982838917] | 3 / 1140 |
| D64 / fwe / 0.8B / 0.0001 / 2030 | 0.010142791 | 0.010140015 | -0.000002777 | full | [0.009461373976680829, 0.010853471373500878] | 0.912784296 | 0.028331805 | 0.35800018671157713 | [0.3386663513674781, 0.3778172507015205] | 3 / 1140 |
| E / fwe / 3.2B / 0.0001 / 2030 | 0.009719015 | 0.009721389 | +0.000002374 | full | [0.009022865073839276, 0.010465821393833934] | 0.101404063 | 0.044961504 | 0.21616302869989215 | [0.2025555849900522, 0.23057547773787312] | 3 / 1140 |
| E64 / fwe / 3.2B / 0.0001 / 2030 | 0.017031452 | 0.017031637 | +0.000000184 | full | [0.016261573587966215, 0.017885384609443087] | 1.014188359 | 0.073293309 | 0.23237390340546138 | [0.22333215481620913, 0.24197638068053337] | 3 / 1140 |
| D64 / fwe / 3.2B / 0.0001 / 2030 | 0.007312437 | 0.007310248 | -0.000002189 | full | [0.006691771781837816, 0.00794790779711956] | 0.912784296 | 0.028331805 | 0.25809994947985004 | [0.23969918789285458, 0.2769041385854895] | 3 / 1140 |

Matched LR (peak/4): 5e-05. OpenSci stable E/G: DCLM 0.51, FWE 0.81; annealed: 0.41, 0.50.

F0 dclm: {'status': 'available', 'nll': 2.7736634966730858, 'parent_P300': 2.7377930322126844, 'reference_minus_parent': 0.03587046446040132, 'break_even': {'budget': None, 'largest_budget': 12.8, 'gap': 0.04299901353590707}, 'nonembedding_parameters': 1205424128, 'reference_pretraining_tokens': 300000000000.0, 'flags': [], 'cross_node_12_8B_note': 'First-node final scalar; cross-node difference about 1e-4'}

| Student | Heal budget | NLL | Student minus 1.3B | Total training tokens |
|---|---|---:|---:|---:|
| P300 | 0.2B | 2.942554079 | +0.168890582 | 300200000000 |
| P1T | 0.2B | 2.912766332 | +0.139102835 | 1000200000000 |
| P300 | 0.8B | 2.879720978 | +0.106057482 | 300800000000 |
| P1T | 0.8B | 2.849623360 | +0.075959864 | 1000800000000 |
| P300 | 3.2B | 2.839891311 | +0.066227815 | 303200000000 |
| P1T | 3.2B | 2.810678156 | +0.037014659 | 1003200000000 |
| P300 | 12.8B | 2.816662510 | +0.042999014 | 312800000000 |
| P1T | 12.8B | 2.786168130 | +0.012504633 | 1012800000000 |

F0 fwe: {'status': 'available', 'nll': 2.893222287314092, 'parent_P300': 2.860631325856815, 'reference_minus_parent': 0.0325909614572768, 'break_even': {'budget': None, 'largest_budget': 3.2, 'gap': 0.06259006519599719}, 'nonembedding_parameters': 1205424128, 'reference_pretraining_tokens': 300000000000.0, 'flags': [], 'cross_node_12_8B_note': 'First-node final scalar; cross-node difference about 1e-4'}

| Student | Heal budget | NLL | Student minus 1.3B | Total training tokens |
|---|---|---:|---:|---:|
| P300 | 0.2B | 3.056353314 | +0.163131027 | 300200000000 |
| P1T | 0.2B | 3.030653544 | +0.137431256 | 1000200000000 |
| P300 | 0.8B | 2.997096324 | +0.103874037 | 300800000000 |
| P1T | 0.8B | 2.967992111 | +0.074769824 | 1000800000000 |
| P300 | 3.2B | 2.955812353 | +0.062590065 | 303200000000 |
| P1T | 3.2B | 2.927349511 | +0.034127223 | 1003200000000 |

Conditional extensions and trigger evidence: {'F3': True, 'F4': True, 'F5': False, 'FE4': {'FE4': 'robust', 'difference': 0.0014928528840014443, 'direction': 'up', 'data_order_difference': 0.0004369966630324712, 'run_F5': False}, 'trigger_records': [{'budget': '0.2B', 'lr': 0.0001, 'seed': 2030, 'data': 'fwe', 'arms': ['k6', 'k20'], 'value': 0.01986042444939224, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 12}, {'budget': '0.8B', 'lr': 0.0001, 'seed': 2030, 'data': 'fwe', 'arms': ['k6', 'k20'], 'value': 0.013324723419155087, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 12}, {'budget': '0.2B', 'lr': 0.0001, 'seed': 2030, 'data': 'fwe', 'arms': ['k20', 'k64'], 'value': 0.014329030549193596, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 16}, {'budget': '0.8B', 'lr': 0.0001, 'seed': 2030, 'data': 'fwe', 'arms': ['k20', 'k64'], 'value': 0.0101427913080627, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 16}, {'budget': '0.2B', 'lr': 0.0001, 'seed': 2030, 'data': 'fwe', 'arms': ['k6', 'k20'], 'value': 0.01986042444939224, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 17}, {'budget': '0.2B', 'lr': 0.0001, 'seed': 2030, 'data': 'mix', 'arms': ['k6', 'k20'], 'value': 0.021353277333393685, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 17}, {'budget': '0.2B', 'lr': 0.0001, 'seed': 2031, 'data': 'fwe', 'arms': ['k6', 'k20'], 'value': 0.02029742111242471, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 17}], 'trigger_policy': 'Primary overlap-adjudicated quantity using evidence available at each registered trigger; records immutable after later runs', 'missing_trigger_policy': 'report unverified and continue independent registered stages; no invented measurement', 'spec_sha256': '40089a98702a285d38799adcdfa89c72c7eec2e75e627cfef8552448f9a593c0'}

Paired document bootstrap: 10,000 resamples, seed 0, token-weighted NLL. Evaluation noise only.
Mixture sampling/source qualifications apply to mixture rows. Shared stable-parent caveat: healing also anneals the student.
Input qualifications: [{'test': 'mixture_source_qualification', 'data': 'mix', 'detail': {'test': 'full_source_uniform_document_sample', 'passed': False, 'detail': 'Uniform documents inside declared shard pools; new corpora use seeded uniform shard selection rather than a global iid document sample. Record finite-pool/cluster-sampling limitation on mixture results; primary FWE unaffected.'}}, {'test': 'FT8_shares_0.2B', 'data': 'mix', 'detail': {'test': 'FT8_shares_0.2B', 'passed': False}}]

Missing: []; invalid: [].
Remaining: verify backup/native handoff and resume Depth.


# Family v1.4 — F6 stable-state extension

Original F0–F5 above retain their v1.3 provenance and scientific flags. F6 uses its own locked v1.4 source.

| Run | Parent | NLL | Δpre | Δpost | Reduced NLL | Reduced Δpost | Flags |
|---|---:|---:|---:|---:|---:|---:|---:|
| ST1-dclm-S188-S1-0.8B-lr1e-3 | 2.886706698 | 2.984526638 | +1.125950111 | +0.097819940 | 2.984526638 | +0.097819940 | 0 |
| ST1-dclm-S72-S1-0.2B-lr1e-3 | 2.908575296 | 3.066636571 | +0.945968926 | +0.158061275 | 3.066636571 | +0.158061275 | 0 |
| ST1-dclm-S72-S1-0.8B-lr1e-3 | 2.908575296 | 2.997767892 | +0.945968926 | +0.089192595 | 2.997767892 | +0.089192595 | 0 |
| ST1-fwe-S188-S1-0.8B-lr1e-3 | 3.023807592 | 3.113324808 | +1.175531177 | +0.089517217 | 3.113467590 | +0.089522732 | 0 |
| ST1-fwe-S188-S1-3.2B-lr1e-3 | 3.023807592 | 3.068752538 | +1.175531177 | +0.044944946 | 3.068890991 | +0.044946134 | 0 |
| ST1-fwe-S72-S1-0.8B-lr1e-3 | 3.042152002 | 3.123058220 | +0.965155970 | +0.080906218 | 3.123197970 | +0.080910756 | 0 |
| ST1-fwe-S72-S1-3.2B-lr1e-3 | 3.042152002 | 3.082068435 | +0.965155970 | +0.039916433 | 3.082204817 | +0.039917604 | 0 |

Full numeric results, CIs, exposure and failed checks follow; no failed row is silently marked verified.

```json
{
  "rows": [
    {
      "run_id": "ST1-dclm-S188-S1-0.8B-lr1e-3",
      "dataset": "dclm",
      "arm": "S188",
      "budget": "0.8B",
      "steps": 3051,
      "tokens": 799801344,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 2.88670669804388,
      "NLL": 2.9845266381346627,
      "delta_pre": 1.1259501105510854,
      "delta_post": 0.09781994009078249,
      "reduced_NLL": 2.9845266381346627,
      "reduced_delta_pre": 1.1259501105510854,
      "reduced_delta_post": 0.09781994009078249,
      "dropped_sequences": 0,
      "dropped_prediction_tokens": 0,
      "exposure": {
        "status": "not applicable; FWE overlap only"
      },
      "flags": []
    },
    {
      "run_id": "ST1-dclm-S72-S1-0.2B-lr1e-3",
      "dataset": "dclm",
      "arm": "S72",
      "budget": "0.2B",
      "steps": 762,
      "tokens": 199753728,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "FT9",
      "parent": 2.908575296107871,
      "NLL": 3.0666365707899006,
      "delta_pre": 0.9459689258930865,
      "delta_post": 0.15806127468202968,
      "reduced_NLL": 3.0666365707899006,
      "reduced_delta_pre": 0.9459689258930865,
      "reduced_delta_post": 0.15806127468202968,
      "dropped_sequences": 0,
      "dropped_prediction_tokens": 0,
      "exposure": {
        "status": "not applicable; FWE overlap only"
      },
      "flags": []
    },
    {
      "run_id": "ST1-dclm-S72-S1-0.8B-lr1e-3",
      "dataset": "dclm",
      "arm": "S72",
      "budget": "0.8B",
      "steps": 3051,
      "tokens": 799801344,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 2.908575296107871,
      "NLL": 2.997767891533186,
      "delta_pre": 0.9459689258930865,
      "delta_post": 0.08919259542531567,
      "reduced_NLL": 2.997767891533186,
      "reduced_delta_pre": 0.9459689258930865,
      "reduced_delta_post": 0.08919259542531567,
      "dropped_sequences": 0,
      "dropped_prediction_tokens": 0,
      "exposure": {
        "status": "not applicable; FWE overlap only"
      },
      "flags": []
    },
    {
      "run_id": "ST1-fwe-S188-S1-0.8B-lr1e-3",
      "dataset": "fwe",
      "arm": "S188",
      "budget": "0.8B",
      "steps": 3051,
      "tokens": 799801344,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 3.0238075916258365,
      "NLL": 3.113324808186919,
      "delta_pre": 1.1755311765990717,
      "delta_post": 0.08951721656108358,
      "reduced_NLL": 3.1134675899283915,
      "reduced_delta_pre": 1.175551583551625,
      "reduced_delta_post": 0.08952273219749085,
      "dropped_sequences": 3,
      "dropped_prediction_tokens": 12285,
      "exposure": {
        "status": "not seen",
        "actual_rows": 195264,
        "documents": [
          {
            "pool_index": 28767024,
            "permutation_position": 1313362,
            "token_start": 1356141489,
            "token_stop": 1356142174,
            "text_sha256": "c15d0766b161a987ef155e8dd455a3859857942a98ee3e10b34b2cd996fcd318",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 685
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 14947840,
            "permutation_position": 6125552,
            "token_start": 6303817369,
            "token_stop": 6303817568,
            "text_sha256": "5baa944c315747108d022c20546c5ad3223c3e913c871f9e5ece7813296aeb78",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 28299105,
            "permutation_position": 17225137,
            "token_start": 17731899194,
            "token_stop": 17731899478,
            "text_sha256": "edb0ff7bea45d20e671713b99c22169de68dc5ba84949be8b648916c5dba9227",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          }
        ]
      },
      "flags": []
    },
    {
      "run_id": "ST1-fwe-S188-S1-3.2B-lr1e-3",
      "dataset": "fwe",
      "arm": "S188",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 3.0238075916258365,
      "NLL": 3.068752537731151,
      "delta_pre": 1.1755311765990717,
      "delta_post": 0.044944946105315346,
      "reduced_NLL": 3.06889099141491,
      "reduced_delta_pre": 1.175551583551625,
      "reduced_delta_post": 0.04494613368400984,
      "dropped_sequences": 3,
      "dropped_prediction_tokens": 12285,
      "exposure": {
        "status": "seen",
        "actual_rows": 781248,
        "documents": [
          {
            "pool_index": 28767024,
            "permutation_position": 1313362,
            "token_start": 1356141489,
            "token_stop": 1356142174,
            "text_sha256": "c15d0766b161a987ef155e8dd455a3859857942a98ee3e10b34b2cd996fcd318",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 685
            },
            "actual_consumed_tokens": 685
          },
          {
            "pool_index": 14947840,
            "permutation_position": 6125552,
            "token_start": 6303817369,
            "token_stop": 6303817568,
            "text_sha256": "5baa944c315747108d022c20546c5ad3223c3e913c871f9e5ece7813296aeb78",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 28299105,
            "permutation_position": 17225137,
            "token_start": 17731899194,
            "token_stop": 17731899478,
            "text_sha256": "edb0ff7bea45d20e671713b99c22169de68dc5ba84949be8b648916c5dba9227",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          }
        ]
      },
      "flags": []
    },
    {
      "run_id": "ST1-fwe-S72-S1-0.8B-lr1e-3",
      "dataset": "fwe",
      "arm": "S72",
      "budget": "0.8B",
      "steps": 3051,
      "tokens": 799801344,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 3.042152001946299,
      "NLL": 3.1230582197769947,
      "delta_pre": 0.9651559695530937,
      "delta_post": 0.08090621783069607,
      "reduced_NLL": 3.1231979698317116,
      "reduced_delta_pre": 0.9652157434331661,
      "reduced_delta_post": 0.08091075634803858,
      "dropped_sequences": 3,
      "dropped_prediction_tokens": 12285,
      "exposure": {
        "status": "not seen",
        "actual_rows": 195264,
        "documents": [
          {
            "pool_index": 28767024,
            "permutation_position": 1313362,
            "token_start": 1356141489,
            "token_stop": 1356142174,
            "text_sha256": "c15d0766b161a987ef155e8dd455a3859857942a98ee3e10b34b2cd996fcd318",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 685
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 14947840,
            "permutation_position": 6125552,
            "token_start": 6303817369,
            "token_stop": 6303817568,
            "text_sha256": "5baa944c315747108d022c20546c5ad3223c3e913c871f9e5ece7813296aeb78",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 28299105,
            "permutation_position": 17225137,
            "token_start": 17731899194,
            "token_stop": 17731899478,
            "text_sha256": "edb0ff7bea45d20e671713b99c22169de68dc5ba84949be8b648916c5dba9227",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          }
        ]
      },
      "flags": []
    },
    {
      "run_id": "ST1-fwe-S72-S1-3.2B-lr1e-3",
      "dataset": "fwe",
      "arm": "S72",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 3.042152001946299,
      "NLL": 3.0820684346236717,
      "delta_pre": 0.9651559695530937,
      "delta_post": 0.03991643267737394,
      "reduced_NLL": 3.082204817023445,
      "reduced_delta_pre": 0.9652157434331661,
      "reduced_delta_post": 0.03991760353977203,
      "dropped_sequences": 3,
      "dropped_prediction_tokens": 12285,
      "exposure": {
        "status": "seen",
        "actual_rows": 781248,
        "documents": [
          {
            "pool_index": 28767024,
            "permutation_position": 1313362,
            "token_start": 1356141489,
            "token_stop": 1356142174,
            "text_sha256": "c15d0766b161a987ef155e8dd455a3859857942a98ee3e10b34b2cd996fcd318",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 685
            },
            "actual_consumed_tokens": 685
          },
          {
            "pool_index": 14947840,
            "permutation_position": 6125552,
            "token_start": 6303817369,
            "token_stop": 6303817568,
            "text_sha256": "5baa944c315747108d022c20546c5ad3223c3e913c871f9e5ece7813296aeb78",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 28299105,
            "permutation_position": 17225137,
            "token_start": 17731899194,
            "token_stop": 17731899478,
            "text_sha256": "edb0ff7bea45d20e671713b99c22169de68dc5ba84949be8b648916c5dba9227",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          }
        ]
      },
      "flags": []
    }
  ],
  "pairs": [
    {
      "dataset": "dclm",
      "budget": "0.8B",
      "full": {
        "E": 0.008627344665466826,
        "E_pre": 0.17998118465799937,
        "G": 0.021868598063990462,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.3945083557812922,
        "bootstrap": {
          "E_ci95": [
            0.008146503158218799,
            0.009101567875214807
          ],
          "E_over_G_ci95": [
            0.37560361078697274,
            0.4137477241031917
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.09672714000895767
      },
      "reduced": {
        "E": 0.008627344665466826,
        "E_pre": 0.17998118465799937,
        "G": 0.021868598063990462,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.3945083557812922,
        "bootstrap": {
          "E_ci95": [
            0.008146503158218799,
            0.009101567875214807
          ],
          "E_over_G_ci95": [
            0.37560361078697274,
            0.4137477241031917
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.09672714000895767
      },
      "difference": 0.0
    },
    {
      "dataset": "fwe",
      "budget": "0.8B",
      "full": {
        "E": 0.008610998730387487,
        "E_pre": 0.21037520704597804,
        "G": 0.018344410320462974,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.469407224323914,
        "bootstrap": {
          "E_ci95": [
            0.008100929041604121,
            0.009123613133210277
          ],
          "E_over_G_ci95": [
            0.4446942800383174,
            0.4946839052677514
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.10643185358641802
      },
      "reduced": {
        "E": 0.008611975849452258,
        "E_pre": 0.21033584011845904,
        "G": 0.018342355752773404,
        "dropped_sequences": 3,
        "dropped_prediction_tokens": 12285,
        "kept_sequences": 2458,
        "kept_prediction_tokens": 10065510,
        "E_over_G": 0.4695130748486387,
        "bootstrap": {
          "E_ci95": [
            0.008096019867176346,
            0.009126679800003918
          ],
          "E_over_G_ci95": [
            0.44441820935624904,
            0.4942767370292942
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.10643796002113416
      },
      "difference": 9.771190647711592e-07
    },
    {
      "dataset": "fwe",
      "budget": "3.2B",
      "full": {
        "E": 0.005028513427941408,
        "E_pre": 0.21037520704597804,
        "G": 0.018344410320462974,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.2741169293587028,
        "bootstrap": {
          "E_ci95": [
            0.004554093812499127,
            0.005517363521021101
          ],
          "E_over_G_ci95": [
            0.251801136453651,
            0.2968443200702413
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.1259760226717792
      },
      "reduced": {
        "E": 0.005028530144237816,
        "E_pre": 0.21033584011845904,
        "G": 0.018342355752773404,
        "dropped_sequences": 3,
        "dropped_prediction_tokens": 12285,
        "kept_sequences": 2458,
        "kept_prediction_tokens": 10065510,
        "E_over_G": 0.2741485451495232,
        "bootstrap": {
          "E_ci95": [
            0.0045502337907834065,
            0.005513036159708115
          ],
          "E_over_G_ci95": [
            0.25082823572612917,
            0.29673741020708644
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.1259727463154852
      },
      "difference": 1.6716296407856546e-08
    }
  ],
  "missing_or_invalid": [],
  "FE6": {
    "dclm": {
      "label": "flat",
      "difference": -0.002504629739437653,
      "unverified": false
    },
    "fwe": {
      "label": "falls",
      "difference": -0.006169745700618403,
      "unverified": false
    }
  },
  "FE6_extended": {
    "fwe": {
      "label": "falls",
      "difference": -0.0035824853024460784
    }
  },
  "E02": {
    "dclm": 0.011131974404904479,
    "fwe": 0.01478074443100589
  },
  "overlap": {
    "immaterial": true,
    "primary_FWE": "full"
  },
  "interval_caveat": "OpenSci native final set is packed, not one document per row; paired sequence CI preserves A2 S and does not claim document-level independence.",
  "events": [
    {
      "name": "ST1-dclm-S72-S1-0.2B-lr1e-3",
      "returncode": 0,
      "seconds": 865.2041508480906,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-dclm-S72-S1-0.2B-lr1e-3.log"
    },
    {
      "name": "ST1-dclm-S72-S1-0.8B-lr1e-3",
      "returncode": 0,
      "seconds": 3063.4453087556176,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-dclm-S72-S1-0.8B-lr1e-3.log"
    },
    {
      "name": "ST1-dclm-S188-S1-0.8B-lr1e-3",
      "returncode": 0,
      "seconds": 3066.0976773388684,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-dclm-S188-S1-0.8B-lr1e-3.log"
    },
    {
      "name": "ST1-fwe-S72-S1-0.8B-lr1e-3",
      "returncode": 0,
      "seconds": 3066.2043864717707,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-fwe-S72-S1-0.8B-lr1e-3.log"
    },
    {
      "name": "ST1-fwe-S188-S1-0.8B-lr1e-3",
      "returncode": 0,
      "seconds": 3062.561924180016,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-fwe-S188-S1-0.8B-lr1e-3.log"
    },
    {
      "name": "ST1-fwe-S72-S1-3.2B-lr1e-3",
      "returncode": 0,
      "seconds": 11860.461530853994,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-fwe-S72-S1-3.2B-lr1e-3.log"
    },
    {
      "name": "ST1-fwe-S188-S1-3.2B-lr1e-3",
      "returncode": 0,
      "seconds": 11863.846811673604,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-fwe-S188-S1-3.2B-lr1e-3.log"
    }
  ],
  "extension": {
    "dataset": "fwe",
    "datasets": {
      "dclm": {
        "E02": 0.011131974404904479,
        "E08": 0.008627344665466826,
        "difference": -0.002504629739437653,
        "threshold": 0.003,
        "relative_change": 0.22499420572995332,
        "trigger": false
      },
      "fwe": {
        "E02": 0.01478074443100589,
        "E08": 0.008610998730387487,
        "difference": -0.006169745700618403,
        "threshold": 0.003,
        "relative_change": 0.4174177917369299,
        "trigger": true
      }
    },
    "tie_policy": "DCLM first for an exact relative-change tie"
  },
  "historical_02B": {
    "dclm": {
      "E_full": 0.011131974404904479,
      "E_reduced": 0.011131974404904479,
      "full": {
        "E": 0.011131974404904479,
        "E_pre": 0.17998118465799937,
        "G": 0.021868598063990462,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.5090392338974279
      },
      "reduced": {
        "E": 0.011131974404904479,
        "E_pre": 0.17998118465799937,
        "G": 0.021868598063990462,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.5090392338974279
      },
      "delta_post": {
        "S72": 0.1584114774050014,
        "S188": 0.1695434518099055
      },
      "origin": "A2 S first node 6y3o; exact saved arrays; no rerun except FT9",
      "E_over_short_delta": 0.07027252436036568
    },
    "fwe": {
      "E_full": 0.01478074443100589,
      "E_reduced": 0.014785743201733266,
      "full": {
        "E": 0.01478074443100589,
        "E_pre": 0.21037520704597804,
        "G": 0.018344410320462974,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.8057355986263642
      },
      "reduced": {
        "E": 0.014785743201733266,
        "E_pre": 0.21033584011845904,
        "G": 0.018342355752773404,
        "dropped_sequences": 3,
        "dropped_prediction_tokens": 12285,
        "kept_sequences": 2458,
        "kept_prediction_tokens": 10065510,
        "E_over_G": 0.8060983769490803
      },
      "delta_post": {
        "S72": 0.1469997437496553,
        "S188": 0.16178048818066193
      },
      "origin": "A2 S first node 6y3o; exact saved arrays; no rerun except FT9",
      "E_over_short_delta": 0.10054945712135331
    }
  },
  "Smol_E_over_short_delta": [
    {
      "budget": "0.2B",
      "E": 0.01986042444939224,
      "short_delta_post": 0.23845270954987827,
      "E_over_short_delta": 0.08328873463791756,
      "primary": "full"
    },
    {
      "budget": "0.8B",
      "E": 0.013324723419155087,
      "short_delta_post": 0.16357947506698883,
      "E_over_short_delta": 0.08145718412226452,
      "primary": "full"
    },
    {
      "budget": "3.2B",
      "E": 0.00971901488669057,
      "short_delta_post": 0.11357928527113147,
      "E_over_short_delta": 0.08557031208190619,
      "primary": "full"
    }
  ],
  "missing_or_failed_runs": [],
  "FT9_pass": true,
  "FT9_flags": [],
  "joint_reading": "mixed",
  "joint_reading_unverified": true,
  "previous_decision": {
    "path": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-20261107-a01/results/a2_family_20261107/decision_family.json",
    "sha256": "9d2ab5b3388247780f270381c0c4ba585a1d66438aac0002b91e4408729557bb"
  },
  "historical_origin": "A2 S on 6y3o; full .2B values reused, not replaced by FT9; node differences retained",
  "FT9_reference": 3.0669867735128715
}
```


# Family v1.7 — F8 and F7

Spec SHA256: `70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209`. Owner-requested extension; original F0–F6 values/readings/flags stand unchanged. FT10 skipped: same wc69 node.

| Run | Parent | NLL | Δpre | Δpost | Flags |
|---|---:|---:|---:|---:|---:|
| ST1-dclm-S188-S1-3.2B-lr1e-3 | 2.886706698 | 2.941622756 | +1.125950111 | +0.054916058 | 0 |
| ST1-dclm-S72-S1-3.2B-lr1e-3 | 2.908575296 | 2.958319402 | +0.945968926 | +0.049744105 | 0 |
| Y3-dclm-P1T-L3-3.2B | 2.686908992 | 2.755552615 | +0.323383519 | +0.068643623 | 0 |
| Y3-dclm-P1T-L9-3.2B | 2.686908992 | 2.871552250 | +3.249153015 | +0.184643258 | 0 |
| Y3-dclm-P300-L3-3.2B | 2.737793032 | 2.789057126 | +0.273158845 | +0.051264094 | 0 |
| Y3-dclm-P300-L9-3.2B | 2.737793032 | 2.897751017 | +2.581002209 | +0.159957985 | 0 |

Paired sequence bootstrap: 10,000 resamples, seed 0; packed rows have 4095 scored targets each. Evaluation noise only. Full addendum quantities, provenance, missing runs and preserved flags:

```json
{
  "F8": {
    "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
    "pairs": {
      "F8-L3": {
        "E": 0.01737952924731561,
        "E_pre": 0.05022467417177427,
        "G": 0.050884040448750244,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.3415516750251005,
        "bootstrap": {
          "E_ci95": [
            0.016971670586036408,
            0.017802500551044386
          ],
          "E_over_G_ci95": [
            0.3347886794692613,
            0.3485257816926494
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "node": "wc69",
        "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
        "flags": [],
        "unverified": false,
        "parent_gain_kept": 0.6584483249748996,
        "E_over_short_delta": 0.33901953410046853
      },
      "F8-L9": {
        "E": 0.0246852738633684,
        "E_pre": 0.668150805442168,
        "G": 0.050884040448750244,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.4851280213927016,
        "bootstrap": {
          "E_ci95": [
            0.024083792822566712,
            0.025277338515471134
          ],
          "E_over_G_ci95": [
            0.4761762078153706,
            0.49406437528951674
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "node": "wc69",
        "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
        "flags": [],
        "unverified": false,
        "parent_gain_kept": 0.5148719786072984,
        "E_over_short_delta": 0.1543234863876951
      }
    },
    "four_cut_table": {
      "6": {
        "E": 0.021670884569823023,
        "G": 0.050884040448750056,
        "parent_gain_kept": 0.5741123468438057,
        "node": "6y3o",
        "origin": "Stage1",
        "source": {
          "path": "resources/a2_depth_stage1_references_20261101.json",
          "sha256": "78afb6090646a84afc59d498911d37590ebce745a8e0a4d637d0411406651da5"
        }
      },
      "12": {
        "E": 0.02332383536097593,
        "G": 0.050884040448750056,
        "parent_gain_kept": 0.5416276861019422,
        "node": "wc69",
        "origin": "Why",
        "historical_flags": [
          [
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 0
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 0
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 1
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 1
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 2
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 2
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 3
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 3
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 4
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 4
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 5
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 5
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 6
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 6
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 7
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 7
            }
          ],
          [
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 0
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 0
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 1
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 1
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 2
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 2
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 3
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 3
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 4
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 4
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 5
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 5
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 6
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 6
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 7
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 7
            }
          ]
        ],
        "qualification": "DCLM cleared by independent Why overlap/row audit; original historical flags retained"
      },
      "3": {
        "E": 0.01737952924731561,
        "E_pre": 0.05022467417177427,
        "G": 0.050884040448750244,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.3415516750251005,
        "bootstrap": {
          "E_ci95": [
            0.016971670586036408,
            0.017802500551044386
          ],
          "E_over_G_ci95": [
            0.3347886794692613,
            0.3485257816926494
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "node": "wc69",
        "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
        "flags": [],
        "unverified": false,
        "parent_gain_kept": 0.6584483249748996,
        "E_over_short_delta": 0.33901953410046853
      },
      "9": {
        "E": 0.0246852738633684,
        "E_pre": 0.668150805442168,
        "G": 0.050884040448750244,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.4851280213927016,
        "bootstrap": {
          "E_ci95": [
            0.024083792822566712,
            0.025277338515471134
          ],
          "E_over_G_ci95": [
            0.4761762078153706,
            0.49406437528951674
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "node": "wc69",
        "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
        "flags": [],
        "unverified": false,
        "parent_gain_kept": 0.5148719786072984,
        "E_over_short_delta": 0.1543234863876951
      }
    },
    "changes": {
      "3": {
        "E08": 0.018639719725659365,
        "E32": 0.01737952924731561,
        "difference": -0.0012601904783437548,
        "label": "flat",
        "direction": "down"
      },
      "9": {
        "E08": 0.022253106709633208,
        "E32": 0.0246852738633684,
        "difference": 0.0024321671537351937,
        "label": "flat",
        "direction": "up"
      }
    },
    "FY3_32B": "mixed",
    "tau": 0.003,
    "secondary": true,
    "original_Why_08_reading_unchanged": "A not supported",
    "slope_per_layer": 0.000694910254484212,
    "E12_minus_E3": 0.005944306113660321,
    "unverified": false,
    "caveat": "DCLM only; within-node E; historical L6 from 6y3o, other cuts wc69. Evaluation noise only, not document independence."
  },
  "F7": {
    "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
    "secondary": true,
    "series": {
      "dclm": {
        "0.2B": {
          "E": 0.011131974404904479,
          "node": "6y3o"
        },
        "0.8B": {
          "E": 0.008627344665466826,
          "E_pre": 0.17998118465799937,
          "G": 0.021868598063990462,
          "dropped_sequences": 0,
          "dropped_prediction_tokens": 0,
          "kept_sequences": 2461,
          "kept_prediction_tokens": 10077795,
          "E_over_G": 0.3945083557812922,
          "bootstrap": {
            "E_ci95": [
              0.008146503158218799,
              0.009101567875214807
            ],
            "E_over_G_ci95": [
              0.37560361078697274,
              0.4137477241031917
            ],
            "resamples": 10000,
            "seed": 0,
            "unit": "paired fixed-length packed sequences (native A2 S estimator)",
            "estimator": "token weighted",
            "zero_token_draws": 0,
            "undefined_ratio_draws": 0
          },
          "E_over_short_delta": 0.09672714000895767,
          "node": "wc69",
          "origin": "F6"
        },
        "3.2B": {
          "E": 0.005171952894106886,
          "E_pre": 0.17998118465799937,
          "G": 0.021868598063990462,
          "dropped_sequences": 0,
          "dropped_prediction_tokens": 0,
          "kept_sequences": 2461,
          "kept_prediction_tokens": 10077795,
          "E_over_G": 0.2365013467700607,
          "bootstrap": {
            "E_ci95": [
              0.004717248259710383,
              0.00562745525299936
            ],
            "E_over_G_ci95": [
              0.21880310598180264,
              0.2540003650582174
            ],
            "resamples": 10000,
            "seed": 0,
            "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
            "estimator": "token weighted",
            "zero_token_draws": 0,
            "undefined_ratio_draws": 0
          },
          "node": "wc69",
          "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
          "flags": [],
          "unverified": false,
          "parent_gain_kept": 0.7634986532299393,
          "E_over_short_delta": 0.10397117098349956
        }
      },
      "fwe": {
        "0.2B": {
          "E": 0.01478074443100589,
          "node": "6y3o"
        },
        "0.8B": {
          "E": 0.008610998730387487,
          "E_pre": 0.21037520704597804,
          "G": 0.018344410320462974,
          "dropped_sequences": 0,
          "dropped_prediction_tokens": 0,
          "kept_sequences": 2461,
          "kept_prediction_tokens": 10077795,
          "E_over_G": 0.469407224323914,
          "bootstrap": {
            "E_ci95": [
              0.008100929041604121,
              0.009123613133210277
            ],
            "E_over_G_ci95": [
              0.4446942800383174,
              0.4946839052677514
            ],
            "resamples": 10000,
            "seed": 0,
            "unit": "paired fixed-length packed sequences (native A2 S estimator)",
            "estimator": "token weighted",
            "zero_token_draws": 0,
            "undefined_ratio_draws": 0
          },
          "E_over_short_delta": 0.10643185358641802,
          "node": "wc69",
          "origin": "F6"
        },
        "3.2B": {
          "E": 0.005028513427941408,
          "E_pre": 0.21037520704597804,
          "G": 0.018344410320462974,
          "dropped_sequences": 0,
          "dropped_prediction_tokens": 0,
          "kept_sequences": 2461,
          "kept_prediction_tokens": 10077795,
          "E_over_G": 0.2741169293587028,
          "bootstrap": {
            "E_ci95": [
              0.004554093812499127,
              0.005517363521021101
            ],
            "E_over_G_ci95": [
              0.251801136453651,
              0.2968443200702413
            ],
            "resamples": 10000,
            "seed": 0,
            "unit": "paired fixed-length packed sequences (native A2 S estimator)",
            "estimator": "token weighted",
            "zero_token_draws": 0,
            "undefined_ratio_draws": 0
          },
          "E_over_short_delta": 0.1259760226717792,
          "node": "wc69",
          "origin": "F6"
        }
      }
    },
    "original_FE6_and_joint_reading_unchanged": true,
    "pair": {
      "E": 0.005171952894106886,
      "E_pre": 0.17998118465799937,
      "G": 0.021868598063990462,
      "dropped_sequences": 0,
      "dropped_prediction_tokens": 0,
      "kept_sequences": 2461,
      "kept_prediction_tokens": 10077795,
      "E_over_G": 0.2365013467700607,
      "bootstrap": {
        "E_ci95": [
          0.004717248259710383,
          0.00562745525299936
        ],
        "E_over_G_ci95": [
          0.21880310598180264,
          0.2540003650582174
        ],
        "resamples": 10000,
        "seed": 0,
        "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
        "estimator": "token weighted",
        "zero_token_draws": 0,
        "undefined_ratio_draws": 0
      },
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
      "flags": [],
      "unverified": false,
      "parent_gain_kept": 0.7634986532299393,
      "E_over_short_delta": 0.10397117098349956
    },
    "FE6_DCLM_32B": {
      "label": "falls",
      "difference": -0.00345539177135994
    },
    "change02_to32": -0.005960021510797593,
    "unverified": false
  },
  "rows": [
    {
      "run_id": "ST1-dclm-S188-S1-3.2B-lr1e-3",
      "arm": "S188",
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "phase": "F7",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.88670669804388,
      "NLL": 2.9416227563970634,
      "delta_pre": 1.1259501105510854,
      "delta_post": 0.05491605835318274,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    },
    {
      "run_id": "ST1-dclm-S72-S1-3.2B-lr1e-3",
      "arm": "S72",
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "phase": "F7",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.908575296107871,
      "NLL": 2.9583194015669463,
      "delta_pre": 0.9459689258930865,
      "delta_post": 0.049744105459075864,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    },
    {
      "run_id": "Y3-dclm-P1T-L3-3.2B",
      "arm": "P1T",
      "removed": [
        19,
        20,
        21
      ],
      "phase": "F8",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.6869089917639344,
      "NLL": 2.755552614962147,
      "delta_pre": 0.32338351898639756,
      "delta_post": 0.06864362319821224,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    },
    {
      "run_id": "Y3-dclm-P1T-L9-3.2B",
      "arm": "P1T",
      "removed": [
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "phase": "F8",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.6869089917639344,
      "NLL": 2.8715522502554713,
      "delta_pre": 3.249153014701919,
      "delta_post": 0.18464325849153734,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    },
    {
      "run_id": "Y3-dclm-P300-L3-3.2B",
      "arm": "P300",
      "removed": [
        19,
        20,
        21
      ],
      "phase": "F8",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.7377930322126844,
      "NLL": 2.789057126163582,
      "delta_pre": 0.2731588448146232,
      "delta_post": 0.05126409395089665,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    },
    {
      "run_id": "Y3-dclm-P300-L9-3.2B",
      "arm": "P300",
      "removed": [
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "phase": "F8",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.7377930322126844,
      "NLL": 2.8977510168408527,
      "delta_pre": 2.5810022092597524,
      "delta_post": 0.15995798462816896,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    }
  ],
  "missing_or_invalid": [],
  "missing_runs": [],
  "coverage_complete": true,
  "events": [
    {
      "name": "Y3-dclm-P300-L3-3.2B",
      "returncode": 0,
      "seconds": 13123.10453401506,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/Y3-dclm-P300-L3-3.2B.log"
    },
    {
      "name": "Y3-dclm-P1T-L3-3.2B",
      "returncode": 0,
      "seconds": 13120.96041374607,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/Y3-dclm-P1T-L3-3.2B.log"
    },
    {
      "name": "Y3-dclm-P300-L9-3.2B",
      "returncode": 0,
      "seconds": 10599.895367615856,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/Y3-dclm-P300-L9-3.2B.log"
    },
    {
      "name": "Y3-dclm-P1T-L9-3.2B",
      "returncode": 0,
      "seconds": 10601.89562015608,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/Y3-dclm-P1T-L9-3.2B.log"
    },
    {
      "name": "ST1-dclm-S72-S1-3.2B-lr1e-3",
      "returncode": 0,
      "seconds": 11871.421059853863,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/ST1-dclm-S72-S1-3.2B-lr1e-3.log"
    },
    {
      "name": "ST1-dclm-S188-S1-3.2B-lr1e-3",
      "returncode": 0,
      "seconds": 11866.271983564831,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/ST1-dclm-S188-S1-3.2B-lr1e-3.log"
    }
  ],
  "pause": {
    "state": "paused",
    "arm": "P1T",
    "saved_state": {
      "path": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-pause-20261109-a01/pause/resume_state.json",
      "sha256": "b5597baa07d79c6bc82e41a5905969759e9375ef55ccf0794a3110ffe10ccc7b"
    },
    "supervisor": {
      "started_at": "2026-11-06T19:20:32.451384+00:00",
      "training_status": "failed",
      "training_error": "KeyboardInterrupt('Supervisor received signal 15')",
      "free_after_training": [
        {
          "index": 0,
          "uuid": "GPU-08f82f92-6e9a-1b21-56cb-4b359c898e3c",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 1,
          "uuid": "GPU-2909c891-748d-a78f-f72c-181231c00e7a",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 2,
          "uuid": "GPU-d5679196-f3b9-f2fd-2a95-fc23676711ba",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 3,
          "uuid": "GPU-8ec230a4-1d3c-13c5-3681-51bceba7df9a",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 4,
          "uuid": "GPU-cdb11306-b5c0-52f0-d871-bba3d94e34d7",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 5,
          "uuid": "GPU-d6a96000-88e7-7653-6d72-9a8e3bddf198",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 6,
          "uuid": "GPU-96e1284a-b396-0253-6511-1d6b2ce193fa",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 7,
          "uuid": "GPU-817e7cb4-510c-c53f-9db1-c5752571469f",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        }
      ],
      "burn": {
        "at": "2026-11-08T17:07:24.755173+00:00",
        "session": "a2_family_20261107_a01_burn",
        "gpus": [
          {
            "index": 0,
            "uuid": "GPU-08f82f92-6e9a-1b21-56cb-4b359c898e3c",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108745
            ]
          },
          {
            "index": 1,
            "uuid": "GPU-2909c891-748d-a78f-f72c-181231c00e7a",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108746
            ]
          },
          {
            "index": 2,
            "uuid": "GPU-d5679196-f3b9-f2fd-2a95-fc23676711ba",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108747
            ]
          },
          {
            "index": 3,
            "uuid": "GPU-8ec230a4-1d3c-13c5-3681-51bceba7df9a",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108748
            ]
          },
          {
            "index": 4,
            "uuid": "GPU-cdb11306-b5c0-52f0-d871-bba3d94e34d7",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108749
            ]
          },
          {
            "index": 5,
            "uuid": "GPU-d6a96000-88e7-7653-6d72-9a8e3bddf198",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108750
            ]
          },
          {
            "index": 6,
            "uuid": "GPU-96e1284a-b396-0253-6511-1d6b2ce193fa",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108751
            ]
          },
          {
            "index": 7,
            "uuid": "GPU-817e7cb4-510c-c53f-9db1-c5752571469f",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108752
            ]
          }
        ],
        "collective_progress_verified": true
      },
      "finished_at": "2026-11-08T17:07:24.755480+00:00"
    },
    "checkpoint": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-pause-20261109-a01/pause/checkpoint-15000",
    "global_step": 15000,
    "handoff_verified": true,
    "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
  }
}
```


---

## Phụ lục B — Depth A: bảng window, correlations và predictions

# A2 Depth — Part A

Spec SHA-256: `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`.

Coverage complete: **True**. Saved arrays: 156/156 (144 pruned + 12 parents).

FP32 parameters rounded to BF16 saved precision, native BF16 AMP/SDPA; final CulturaX set, unchanged A2 scorer.

Paired sequence bootstrap: 10,000 draws, seed 0. Relative intervals recompute the reference damage in every paired draw.

| Dataset | Set | L_ann / D(P300) | L_stab / D(S72) | R_short / D(S72) | R_long / D(S188) |
|---|---|---:|---:|---:|---:|
| dclm | W1 | -0.231621 | -0.146985 | 0.141551 | 0.028286 |
| dclm | W2 | 0.047056 | 0.065850 | 0.605570 | 0.577259 |
| dclm | W3 | 0.245461 | -0.100480 | 0.409317 | 0.951317 |
| dclm | W4 | 0.088436 | 0.090806 | 0.036337 | 0.034086 |
| dclm | W5 | -0.007412 | 0.010100 | 0.047619 | 0.029456 |
| dclm | W6 | 0.066162 | 0.038535 | 0.139724 | 0.170043 |
| dclm | W7 | 0.229747 | 0.161430 | 0.194582 | 0.264850 |
| dclm | W8 | 0.284440 | 0.190261 | 0.084583 | 0.170400 |
| dclm | W9 | 0.289559 | 0.127370 | 0.121081 | 0.282366 |
| dclm | S2 | -0.231621 | -0.146985 | 0.141551 | 0.028286 |
| dclm | S3 | 0.140521 | 0.001845 | 0.148428 | 0.307394 |
| dclm | S4 | 4.008777 | 0.203851 | 0.806788 | 6.517374 |
| fwe | W1 | -0.296443 | 1.184652 | 1.007967 | -0.353344 |
| fwe | W2 | -0.160492 | -0.340845 | 0.342944 | 0.710391 |
| fwe | W3 | 0.075268 | -0.028993 | 0.333506 | 0.476691 |
| fwe | W4 | 1.294981 | 1.022075 | -0.003822 | 0.130625 |
| fwe | W5 | 0.485164 | 0.530033 | -0.014421 | -0.043323 |
| fwe | W6 | 0.398667 | 0.314279 | 0.044520 | 0.111587 |
| fwe | W7 | 0.357375 | 0.223246 | 0.136323 | 0.260920 |
| fwe | W8 | 0.368472 | 0.217970 | -0.003418 | 0.119728 |
| fwe | W9 | 0.239371 | 0.052643 | 0.142567 | 0.345247 |
| fwe | S2 | -0.296443 | 1.184652 | 1.007967 | -0.353344 |
| fwe | S3 | 0.169053 | 0.297311 | 0.269707 | 0.144178 |
| fwe | S4 | 0.909501 | 2.728301 | 1.035943 | 0.042736 |
| nem | W1 | -0.225489 | -0.519324 | 0.118515 | 0.802257 |
| nem | W2 | -0.158909 | -0.109273 | 1.149663 | 1.029871 |
| nem | W3 | 0.498893 | 0.144377 | 1.304653 | 2.018610 |
| nem | W4 | 0.058023 | 0.149579 | 0.052429 | -0.031389 |
| nem | W5 | -0.039524 | -0.066119 | 0.044113 | 0.073847 |
| nem | W6 | 0.076071 | 0.011021 | 0.081790 | 0.151393 |
| nem | W7 | 0.169799 | 0.200485 | 0.209834 | 0.178909 |
| nem | W8 | 0.275450 | 0.349028 | 0.213056 | 0.146895 |
| nem | W9 | 0.309117 | 0.152429 | 0.128916 | 0.282408 |
| nem | S2 | 0.059622 | 0.266550 | 1.657121 | 1.223004 |
| nem | S3 | -0.478140 | -0.467518 | 0.307518 | 0.281437 |
| nem | S4 | -0.104946 | 0.274928 | 1.772607 | 0.946488 |

Full absolute/relative estimates, intervals, damages, correlations and every prediction clause are in `decision_depth.json`.

Spec defines contrasts in nats and additionally requests relative values. Both clause evaluations are reported explicitly, without choosing a scale after observing results.

Predictions (nats): `{"A1": false, "A2": false, "A3": false, "A4": false}`.

Predictions (relative): `{"A1": false, "A2": false, "A3": false, "A4": false}`.

Scientific flags: 0. Failed checks retain their original values; remaining registered work continues.

Deviations: ["Breadth completed before Depth A/B because Depth spec was unavailable; historical ordering deviation preserved."].

Ledger: see `ledger.json`; estimates and elapsed time are report-only.

Remaining: DT4, Part B and its report, DT6, Part C, downstream, independent replay and native handoff.


## Phụ lục C — Depth B: bảng W1/W9 cạnh S1 và clauses

# A2 Depth — Parts A and B

Spec SHA-256: `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`.

Part A source report: {'path': '/mnt/local/_outputs/deep-llms_th2/6y3o-a2-depth-parta-20261101-a01/results/a2_depth_20261031/REPORT.md', 'sha256': '6ae54a45836d2aa670824ac01b598bcac30754b4ad774d59519f0869952b16b5'}.

| Dataset | Block | E | E / Δ(P300) | A_init | A_init / Δ(P300) | E label |
|---|---|---:|---:|---:|---:|---|
| dclm | W1 | 0.028632244673585383 | 0.21539858152781002 | 0.0712501955104603 | 0.5360107536623143 | {'value': 0.028632244673585383, 'clauses': {'INCOMPLETE': False, 'persists': True, 'reversed': False, 'erased': False}, 'label': 'persists'} |
| dclm | W9 | 0.03303367332921914 | 0.11427491360908006 | 0.07713408873095728 | 0.2668335197905046 | {'value': 0.03303367332921914, 'clauses': {'INCOMPLETE': False, 'persists': True, 'reversed': False, 'erased': False}, 'label': 'persists'} |
| dclm | S1 | 0.021096293347514283 | 0.10302884126700326 | 0.07272470370653528 | 0.35516864649842933 | persists |
| fwe | W1 | 0.0345100244148524 | 0.2812193407371629 | 0.07505336227155057 | 0.611603654763835 | {'value': 0.0345100244148524, 'clauses': {'INCOMPLETE': False, 'persists': True, 'reversed': False, 'erased': False}, 'label': 'persists'} |
| fwe | W9 | 0.039414471399715545 | 0.15368555952561244 | 0.08178711849348197 | 0.318905686700369 | {'value': 0.039414471399715545, 'clauses': {'INCOMPLETE': False, 'persists': True, 'reversed': False, 'erased': False}, 'label': 'persists'} |
| fwe | S1 | 0.026224221566386152 | 0.1339870997016109 | 0.07847837360805077 | 0.4009686099708777 | persists |

Predictions: `{"B1": true, "B2": false}`. Every clause: `{"dclm": {"B1": true, "B2": false}, "fwe": {"B1": true, "B2": false}}`.

Scientific coverage complete: True. Flags: 0.

DT4 and scientific flags remain recorded; they do not cancel Part C. Ledger: ledger.json (report-only).

Remaining: publish this A+B result, DT6, three 12.8B fits, downstream, independent replay and final native handoff.

## Part A: completed depth table and A1–A4

The Part A numerical results below are retained from its earlier report; its then-pending-stage footer is omitted here.


Spec SHA-256: `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`.

Coverage complete: **True**. Saved arrays: 156/156 (144 pruned + 12 parents).

FP32 parameters rounded to BF16 saved precision, native BF16 AMP/SDPA; final CulturaX set, unchanged A2 scorer.

Paired sequence bootstrap: 10,000 draws, seed 0. Relative intervals recompute the reference damage in every paired draw.

| Dataset | Set | L_ann / D(P300) | L_stab / D(S72) | R_short / D(S72) | R_long / D(S188) |
|---|---|---:|---:|---:|---:|
| dclm | W1 | -0.231621 | -0.146985 | 0.141551 | 0.028286 |
| dclm | W2 | 0.047056 | 0.065850 | 0.605570 | 0.577259 |
| dclm | W3 | 0.245461 | -0.100480 | 0.409317 | 0.951317 |
| dclm | W4 | 0.088436 | 0.090806 | 0.036337 | 0.034086 |
| dclm | W5 | -0.007412 | 0.010100 | 0.047619 | 0.029456 |
| dclm | W6 | 0.066162 | 0.038535 | 0.139724 | 0.170043 |
| dclm | W7 | 0.229747 | 0.161430 | 0.194582 | 0.264850 |
| dclm | W8 | 0.284440 | 0.190261 | 0.084583 | 0.170400 |
| dclm | W9 | 0.289559 | 0.127370 | 0.121081 | 0.282366 |
| dclm | S2 | -0.231621 | -0.146985 | 0.141551 | 0.028286 |
| dclm | S3 | 0.140521 | 0.001845 | 0.148428 | 0.307394 |
| dclm | S4 | 4.008777 | 0.203851 | 0.806788 | 6.517374 |
| fwe | W1 | -0.296443 | 1.184652 | 1.007967 | -0.353344 |
| fwe | W2 | -0.160492 | -0.340845 | 0.342944 | 0.710391 |
| fwe | W3 | 0.075268 | -0.028993 | 0.333506 | 0.476691 |
| fwe | W4 | 1.294981 | 1.022075 | -0.003822 | 0.130625 |
| fwe | W5 | 0.485164 | 0.530033 | -0.014421 | -0.043323 |
| fwe | W6 | 0.398667 | 0.314279 | 0.044520 | 0.111587 |
| fwe | W7 | 0.357375 | 0.223246 | 0.136323 | 0.260920 |
| fwe | W8 | 0.368472 | 0.217970 | -0.003418 | 0.119728 |
| fwe | W9 | 0.239371 | 0.052643 | 0.142567 | 0.345247 |
| fwe | S2 | -0.296443 | 1.184652 | 1.007967 | -0.353344 |
| fwe | S3 | 0.169053 | 0.297311 | 0.269707 | 0.144178 |
| fwe | S4 | 0.909501 | 2.728301 | 1.035943 | 0.042736 |
| nem | W1 | -0.225489 | -0.519324 | 0.118515 | 0.802257 |
| nem | W2 | -0.158909 | -0.109273 | 1.149663 | 1.029871 |
| nem | W3 | 0.498893 | 0.144377 | 1.304653 | 2.018610 |
| nem | W4 | 0.058023 | 0.149579 | 0.052429 | -0.031389 |
| nem | W5 | -0.039524 | -0.066119 | 0.044113 | 0.073847 |
| nem | W6 | 0.076071 | 0.011021 | 0.081790 | 0.151393 |
| nem | W7 | 0.169799 | 0.200485 | 0.209834 | 0.178909 |
| nem | W8 | 0.275450 | 0.349028 | 0.213056 | 0.146895 |
| nem | W9 | 0.309117 | 0.152429 | 0.128916 | 0.282408 |
| nem | S2 | 0.059622 | 0.266550 | 1.657121 | 1.223004 |
| nem | S3 | -0.478140 | -0.467518 | 0.307518 | 0.281437 |
| nem | S4 | -0.104946 | 0.274928 | 1.772607 | 0.946488 |

Full absolute/relative estimates, intervals, damages, correlations and every prediction clause are in `decision_depth.json`.

Spec defines contrasts in nats and additionally requests relative values. Both clause evaluations are reported explicitly, without choosing a scale after observing results.

Predictions (nats): `{"A1": false, "A2": false, "A3": false, "A4": false}`.

Predictions (relative): `{"A1": false, "A2": false, "A3": false, "A4": false}`.

Scientific flags: 0. Failed checks retain their original values; remaining registered work continues.

Deviations: ["Breadth completed before Depth A/B because Depth spec was unavailable; historical ordering deviation preserved."].

Ledger: see `ledger.json`; estimates and elapsed time are report-only.
