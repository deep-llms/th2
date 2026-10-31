# A2 M — Kết quả mechanism probes, 31/10/2026

**HOÀN TẤT.** GPU probes kết thúc **17:37:54 SGT**; CPU readout hoàn tất **17:40:24 SGT**. Đã xác minh native handoff, tải đủ 578 file JSON/báo cáo/receipt về máy phát triển và tính lại các thống kê cục bộ. Không còn bước tính toán đã đăng ký nào đang chờ.

Spec v1.0 SHA-256: `82093a3fa9ae6d9428c007e8d26c112d875dae0b84dfe83eee3677cf4c8184d4`.

## Kết quả theo tiêu chí đăng ký trước

| Giả thuyết | Đạt tiêu chí? | Kết quả quyết định |
|---|---|---|
| H-P1: lớp ít dư thừa hơn khi train lâu | Có | Tổng damage của các lớp S1 riêng lẻ có pruning pattern trên cả ba dataset. BI_S1 riêng không đạt toàn bộ pattern. |
| H-P2: mức phụ thuộc giữa các lớp tăng | Không | Với ρ, Nemotron có L_ann nhỏ hơn R_short. |
| H-Q1: annealing làm sai số lượng tử hóa trọng số lớn hơn | Không | RTN-3 weight error có R_short nhỏ hơn L_stab trên cả ba dataset. |
| H-Q2: annealing làm model nhạy hơn với cùng mức noise tương đối | Có | N-iso(0.2) có R_short dương và lớn hơn L_stab trên cả ba dataset. |
| H-E: phần lợi thế bị mất tập trung ở token hiếm hoặc khó | Không | Điều kiện token hiếm không đạt ở cả DCLM và FWE; retention của ba bucket hiếm cao hơn ba bucket phổ biến. |

Các kết luận trên là association giữa checkpoint công khai, không chứng minh quan hệ nhân quả. “Không đạt” nghĩa là không đáp ứng toàn bộ tiêu chí đã đăng ký; không chứng minh cơ chế không tồn tại.

## Relative contrasts

Mỗi giá trị là contrast chia metric của parent tham chiếu. Noise dùng trung bình hai seed.

| Metric | Dataset | L_ann | L_stab | R_short |
|---|---|---|---|---|
| S1_sum | dclm | +19.930% | +13.788% | +11.026% |
| S1_sum | fwe | +20.637% | +14.703% | +11.243% |
| S1_sum | nem | +17.796% | +18.208% | +9.068% |
| BI_S1 | dclm | +19.460% | +9.340% | +10.838% |
| BI_S1 | fwe | +29.504% | +23.825% | +18.062% |
| BI_S1 | nem | +242.158% | +371.277% | +14.084% |
| rho | dclm | +8.486% | +5.810% | -1.545% |
| rho | fwe | +13.912% | +6.995% | -9.932% |
| rho | nem | +8.011% | +13.100% | +11.477% |
| rtn3_weight_error | dclm | +3.098% | +1.757% | +0.196% |
| rtn3_weight_error | fwe | +2.967% | +1.713% | +0.190% |
| rtn3_weight_error | nem | +3.118% | +1.707% | +0.122% |
| N-iso-0.2 | dclm | +2.731% | -7.617% | +18.372% |
| N-iso-0.2 | fwe | +12.487% | +1.865% | +19.753% |
| N-iso-0.2 | nem | -2.168% | -19.750% | +14.750% |

## Token retention

Bucket frequency dùng số lần token type xuất hiện trên **toàn bộ 4096 vị trí** của final set; NLL được chấm tại vị trí 1–4095. Difficulty dùng NLL của S188. Các CI dùng paired bootstrap 10.000 lần, seed 0, đơn vị sequence.

| Dataset / loại bucket | Mean retention ba bucket thấp nhất | Ba bucket cao nhất | CI95 của thấp − cao |
|---|---|---|---|
| dclm / frequency | 0.580935 | 0.527968 | [0.021485, 0.084796] |
| dclm / difficulty | 0.766001 | 0.514517 | [0.189953, 0.313066] |
| fwe / frequency | 0.567081 | 0.519037 | [0.018713, 0.077965] |
| fwe / difficulty | 0.496565 | 0.477390 | [-0.296809, 0.202276] |

Frequency thấp = hiếm; difficulty thấp = dễ. Điều kiện difficulty đạt về điểm ước lượng ở cả hai dataset, nhưng CI FWE bao gồm 0. Bucket difficulty số 0 ở cả hai dataset có |parent gap| < 0.005; giữ cờ mẫu số nhỏ và không diễn giải tỷ số đó như ước lượng ổn định.

## Kiểm chứng và các sai lệch được giữ lại

- Đủ 12 parent cho CPU weights, layers và noise; 168 matrices/parent, 24 single-layer bypass/parent, 96 noise cases và 18 per-token arrays. Các baseline/pre-S1/sigma-zero được lưu riêng.
- 210 worker checks và 18 external MT6 replay PASS. Toàn bộ tests.json có 286 mục: 285 PASS và một cờ lỗi điều phối gốc được giữ lại. Canonical CPU readout báo coverage đầy đủ và hai lần tính lại khớp hoàn toàn. Kiểm chứng cục bộ xác nhận 462 per-sequence means (sai lệch lớn nhất 3.552713678800501e-15), 2016 dòng matrix, 747 relative contrasts; paired bootstrap noise 10.000 lần/seed 0 khớp tuyệt đối.
- **Lỗi điều phối gốc được giữ nguyên:** `progress.json.tmp` bị hai luồng ghi đồng thời; future CPU báo FileNotFoundError khi coordinator thu kết quả, nên coordinator gốc exit 1 sau khi các GPU probes đã hoàn tất. Luồng CPU bổ sung hoàn tất đủ weights và readout, returncode 0. Không có GPU rerun, training hay compression mới.
- Readout canonical dùng source CPU `2661b7873ae526344fbdeb00599ff2e2e1945760`; frozen GPU source `1e865abe929563d4f6525e6b95383cf55bebf282` giữ nguyên. Sửa cách đếm frequency để tính cả vị trí 0; không thay đổi NLL đã lưu.
- QT3 cũ và saved-grid revision của A2 Q được giữ trong báo cáo A2 Q; QT6 là minor deviation đã được chấp nhận. Calibration sensitivity của GPTQ vẫn được giữ khi đọc các correlation liên quan.
- Native supervisor bàn giao GPU lúc **17:39:05 SGT**. Kiểm tra độc lập lúc **17:52:27–17:52:39 SGT** xác nhận tám burn worker đúng ownership, cycles tăng 940 → 950; guard đã nhả.

## Compute ledger

GPU probes bắt đầu **16:14:38 SGT**, kết thúc **17:37:54 SGT**: khoảng **1.388 node-hour** (4995.64 giây). Tính tới native handoff là **1.407 node-hour**. Canonical CPU readout mất **71.70 giây**, thực hiện sau handoff; nhánh CPU weights chạy chồng thời gian với GPU, không cộng trùng 4363.67 giây vòng đời của nhánh này vào GPU ledger. Từ lúc supervisor bắt đầu chuẩn bị (16:13:37 SGT) tới CPU readout hoàn tất là **1.446233 giờ** theo đồng hồ, tính các phần chạy song song một lần. Vượt dự toán khoảng 1 node-hour đã đăng ký; thời gian là report-only, không phải tiêu chí khoa học.

## Bằng chứng và trạng thái lưu kết quả

Receipt terminal: [final/verification.json](final/verification.json). Raw receipt SHA-256: `8931fcaa21201f2c8ac79da494c0debd6f627283c03ee58a86616cf45c93e0bb`.

Các bảng canonical đã tải và kiểm tra SHA:

- [weights.json](weights.json), [layers.json](layers.json), [noise_paired.json](noise_paired.json).
- [buckets.json](buckets.json), [decision_mech.json](decision_mech.json), [tests.json](tests.json).
- [token_inventory.json](token_inventory.json), [verification.json](verification.json), [config/plan.json](config/plan.json).
- [Replay trên máy phát triển](final/local_replay.json) và [script tái kiểm chứng](final/replay_download.py).
- [Manifest 578 file](artifacts/manifest.json); per-sequence files nằm trong `artifacts/node/results/a2_mech_20261031/workers/`. Bundle SHA-256: `ee7cc0d1e7edc4f7e45e5ba8421a3bbef99a653ce967f4267fa9ab00aafb9f49`.

Toàn bộ bảng và per-sequence files có ở máy này, giữ ngoài Git để đáp ứng giới hạn code sync 25 MiB. Báo cáo, receipt và script tái kiểm chứng được lưu trong Git. **18 per-token arrays đã được sao lưu khỏi B200 về máy này**, tổng **725.603.544 byte**. Các file `.npy` nằm trong [artifacts/token_arrays/](artifacts/token_arrays/); bản gốc trên B200 được giữ nguyên. Toàn bộ 18 checksum khớp manifest gốc, dtype `float32`, shape `(2461, 4095)`; NLL trung bình theo sequence khớp metadata với sai lệch tối đa **0**. [Receipt sao lưu và đường dẫn từng file](final/token_backup_verification.json), [manifest các phần truyền](final/token_backup_manifest.json). Receipt replay cục bộ ban đầu ghi `token_arrays_downloaded=false` là snapshot lịch sử trước khi sao lưu; receipt mới này xác nhận hoàn tất.

Không còn bước đã đăng ký chờ thực hiện. Cờ còn giữ: lỗi điều phối gốc, bucket mẫu số nhỏ và các hạn chế diễn giải nêu trên.

<!-- A2_M_ADDENDUM_START -->
## Phụ lục theo yêu cầu — bảng đầy đủ từ JSON đã lưu

Bổ sung bằng CPU từ các kết quả canonical đã khóa SHA; không chạy lại model, quantization hoặc GPU. [Kiểm chứng các bảng phụ lục](final/addendum_verification.json) và [script tạo bảng](final/report_addendum.py). Các điểm noise là trung bình seed 0 và 1. Damage/gap/A_init có đơn vị nats/token; retention và correlation là số không thứ nguyên. Giá trị bảng làm tròn để đọc; JSON giữ đầy đủ độ chính xác.

`L_ann = (P1T − P300) / P300`, `L_stab = (S188 − S72) / S72`, `R_short = (P300 − S72) / S72`, áp dụng riêng cho metric đang xét. Các relative contrasts hiển thị dưới dạng %. Mẫu số là giá trị metric của parent tham chiếu, không phải NLL parent.

### 1. N-rtn3: damage từng parent và tỷ số so với RTN-3

| Dataset | Parent | N-rtn3 damage | RTN-3 damage | N-rtn3 / RTN-3 |
|---|---|---|---|---|
| dclm | S72 | 0.053240 | 0.054708 | 0.973165 |
| dclm | P300 | 0.068578 | 0.065746 | 1.043081 |
| dclm | S188 | 0.064197 | 0.047862 | 1.341288 |
| dclm | P1T | 0.077539 | 0.076930 | 1.007913 |
| fwe | S72 | 0.048593 | 0.039110 | 1.242462 |
| fwe | P300 | 0.059360 | 0.054330 | 1.092581 |
| fwe | S188 | 0.050629 | 0.046879 | 1.079982 |
| fwe | P1T | 0.070935 | 0.056925 | 1.246117 |
| nem | S72 | 0.056674 | 0.041329 | 1.371279 |
| nem | P300 | 0.063750 | 0.058321 | 1.093089 |
| nem | S188 | 0.048437 | 0.041934 | 1.155091 |
| nem | P1T | 0.071086 | 0.064363 | 1.104462 |

**Relative contrasts của N-rtn3 damage:**

| Dataset | L_ann | L_stab | R_short |
|---|---|---|---|
| dclm | +13.066% | +20.581% | +28.811% |
| fwe | +19.499% | +4.190% | +22.158% |
| nem | +11.508% | -14.532% | +12.486% |

### 2. N-iso(0.1) và N-iso(0.3)

**N-iso-0.1**

| Dataset | L_ann | L_stab | R_short |
|---|---|---|---|
| dclm | +7.051% | -2.617% | +20.683% |
| fwe | +7.807% | -3.197% | +18.599% |
| nem | -3.819% | -20.619% | +13.800% |

**N-iso-0.3**

| Dataset | L_ann | L_stab | R_short |
|---|---|---|---|
| dclm | +6.012% | -5.578% | +17.890% |
| fwe | +12.787% | +1.762% | +21.129% |
| nem | -1.499% | -20.651% | +16.231% |

Các ordering flips giữa hai seed được giữ dưới đây; `đổi` nghĩa là mệnh đề có truth value khác giữa seed 0 và 1, không phải điểm trung bình không xác định.

| Probe | Dataset | Mệnh đề đổi giữa seed 0 và 1 |
|---|---|---|
| N-rtn3 | dclm | L_stab_above_R_short, R_short_above_L_stab |
| N-rtn3 | fwe | Không |
| N-rtn3 | nem | L_ann_above_R_short |
| N-iso-0.1 | dclm | L_stab_positive |
| N-iso-0.1 | fwe | Không |
| N-iso-0.1 | nem | Không |
| N-iso-0.3 | dclm | Không |
| N-iso-0.3 | fwe | Không |
| N-iso-0.3 | nem | L_ann_positive |

### 3. Correlation sau khi trừ mean từng dataset

Với mỗi metric và mỗi loại damage, trừ mean riêng của bốn parent trong từng dataset, rồi tính Pearson r trên 12 residual (3 dataset × 4 parent). Không có p-value hoặc suy luận nhân quả. GPTQ-3 dùng cal0 theo JSON đã đăng ký; cờ calibration sensitivity của A2 Q vẫn áp dụng.

| Metric | r với pre-S1 damage | r với rtn3 damage | r với gptq3 damage |
|---|---|---|---|
| excess_kurtosis | 0.266346 | 0.816137 | 0.775553 |
| group_range_over_rms | 0.962173 | 0.489471 | 0.746941 |
| rtn3_weight_error | 0.963512 | 0.506069 | 0.759535 |
| rtn4_weight_error | 0.965302 | 0.529416 | 0.778352 |
| BI_S1 | 0.620055 | 0.115579 | 0.185490 |
| S1_sum | 0.950564 | 0.693840 | 0.859087 |
| rho | 0.818827 | 0.251051 | 0.424166 |
| BI_0 | 0.409205 | 0.646640 | 0.670187 |
| BI_1 | 0.737921 | 0.384406 | 0.409730 |
| BI_2 | 0.928577 | 0.527311 | 0.659706 |
| BI_3 | 0.947514 | 0.611181 | 0.741848 |
| BI_4 | 0.955549 | 0.621661 | 0.764218 |
| BI_5 | 0.964410 | 0.634942 | 0.784944 |
| BI_6 | 0.941399 | 0.729166 | 0.822385 |
| BI_7 | 0.941933 | 0.659202 | 0.798574 |
| BI_8 | 0.940105 | 0.713372 | 0.816183 |
| BI_9 | 0.934341 | 0.750333 | 0.830246 |
| BI_10 | 0.904919 | 0.766135 | 0.810431 |
| BI_11 | 0.904402 | 0.751143 | 0.779113 |
| BI_12 | 0.885668 | 0.749185 | 0.779739 |
| BI_13 | 0.849846 | 0.794436 | 0.789518 |
| BI_14 | 0.855177 | 0.831435 | 0.811070 |
| BI_15 | 0.910210 | 0.816191 | 0.831546 |
| BI_16 | 0.916812 | 0.794033 | 0.828687 |
| BI_17 | 0.928630 | 0.767214 | 0.810115 |
| BI_18 | 0.958093 | 0.721341 | 0.837165 |
| BI_19 | 0.929587 | 0.728783 | 0.815542 |
| BI_20 | 0.771118 | 0.369671 | 0.428787 |
| BI_21 | 0.521324 | -0.017427 | 0.062657 |
| BI_22 | 0.789472 | 0.724197 | 0.721586 |
| BI_23 | 0.945189 | 0.534135 | 0.804180 |
| update_0 | 0.748889 | 0.071290 | 0.474724 |
| update_1 | -0.824597 | -0.830218 | -0.915558 |
| update_2 | -0.866851 | -0.674660 | -0.872688 |
| update_3 | 0.179798 | -0.481209 | -0.302523 |
| update_4 | 0.922667 | 0.513540 | 0.647617 |
| update_5 | 0.964392 | 0.606717 | 0.767275 |
| update_6 | 0.560329 | 0.746625 | 0.581490 |
| update_7 | 0.828735 | 0.510552 | 0.667739 |
| update_8 | 0.943181 | 0.666609 | 0.821949 |
| update_9 | 0.946390 | 0.651177 | 0.770034 |
| update_10 | 0.922239 | 0.736595 | 0.794546 |
| update_11 | 0.922471 | 0.715720 | 0.756908 |
| update_12 | 0.845540 | 0.579682 | 0.674301 |
| update_13 | 0.870252 | 0.778486 | 0.802453 |
| update_14 | 0.879417 | 0.811046 | 0.827481 |
| update_15 | 0.906477 | 0.803209 | 0.838370 |
| update_16 | 0.901294 | 0.820825 | 0.850885 |
| update_17 | 0.861399 | 0.859836 | 0.853455 |
| update_18 | 0.845196 | 0.866610 | 0.889549 |
| update_19 | 0.607259 | 0.971572 | 0.873532 |
| update_20 | 0.684680 | 0.778839 | 0.676054 |
| update_21 | 0.583231 | 0.003627 | 0.122258 |
| update_22 | 0.080200 | 0.064975 | -0.106478 |
| update_23 | 0.938567 | 0.448451 | 0.730569 |
| single_damage_0 | -0.377906 | 0.050695 | -0.255468 |
| single_damage_1 | 0.606973 | 0.158273 | 0.241914 |
| single_damage_2 | 0.822804 | 0.137318 | 0.492782 |
| single_damage_3 | 0.901164 | 0.315841 | 0.637663 |
| single_damage_4 | 0.846719 | 0.173758 | 0.536887 |
| single_damage_5 | 0.826279 | 0.346877 | 0.666369 |
| single_damage_6 | 0.919831 | 0.484042 | 0.715066 |
| single_damage_7 | 0.830926 | 0.287847 | 0.515009 |
| single_damage_8 | 0.576875 | 0.063977 | 0.188543 |
| single_damage_9 | 0.814681 | 0.743162 | 0.788504 |
| single_damage_10 | 0.614618 | 0.756496 | 0.783560 |
| single_damage_11 | 0.445420 | 0.163630 | 0.202939 |
| single_damage_12 | 0.564748 | 0.273582 | 0.359985 |
| single_damage_13 | 0.708187 | 0.755153 | 0.787495 |
| single_damage_14 | 0.823567 | 0.770525 | 0.853488 |
| single_damage_15 | 0.768308 | 0.827522 | 0.874205 |
| single_damage_16 | 0.858061 | 0.806932 | 0.903460 |
| single_damage_17 | 0.918628 | 0.759829 | 0.882049 |
| single_damage_18 | 0.918832 | 0.721988 | 0.898295 |
| single_damage_19 | 0.931092 | 0.749846 | 0.907666 |
| single_damage_20 | 0.929630 | 0.734309 | 0.926542 |
| single_damage_21 | 0.776430 | 0.335001 | 0.481112 |
| single_damage_22 | 0.408364 | 0.935820 | 0.770750 |
| single_damage_23 | 0.728427 | 0.935239 | 0.899336 |
| N-iso-0.1 | 0.287997 | 0.859082 | 0.753320 |
| N-iso-0.2 | 0.318506 | 0.872447 | 0.743998 |
| N-iso-0.3 | 0.352728 | 0.880167 | 0.771404 |
| N-rtn3 | 0.621938 | 0.841650 | 0.862247 |

### 4. Profile single-layer bypass: đủ 24 lớp

Chỉ số lớp bắt đầu từ 0; S1 là lớp 16–21. Bảng dùng damage của việc bypass riêng từng lớp trên small set. Các giá trị relative có thể lớn khi damage parent tham chiếu nhỏ; không cộng hoặc lấy trung bình các tỷ lệ này để suy ra damage của cả block.

**dclm**

| Lớp | Thuộc S1 (16–21) | L_ann | L_stab | R_short |
|---|---|---|---|---|
| 0 | — | -54.776% | -29.759% | +23.602% |
| 1 | — | +43.296% | +8.439% | -30.913% |
| 2 | — | +123.174% | +60.613% | -20.028% |
| 3 | — | +94.878% | +35.749% | -16.680% |
| 4 | — | +44.685% | +27.595% | -8.170% |
| 5 | — | +36.321% | +24.997% | +4.969% |
| 6 | — | +17.682% | +16.813% | +8.697% |
| 7 | — | +26.167% | +19.339% | +5.782% |
| 8 | — | +9.870% | +14.414% | +6.839% |
| 9 | — | +11.818% | +7.844% | +4.691% |
| 10 | — | +2.150% | +6.980% | +8.144% |
| 11 | — | +4.458% | +10.357% | +6.071% |
| 12 | — | -1.319% | +3.383% | +9.479% |
| 13 | — | +6.892% | +13.079% | +10.063% |
| 14 | — | +10.432% | +13.764% | +12.420% |
| 15 | — | +7.042% | +10.184% | +13.636% |
| 16 | Có | +11.507% | +10.200% | +11.257% |
| 17 | Có | +15.229% | +13.561% | +11.066% |
| 18 | Có | +23.201% | +18.889% | +11.284% |
| 19 | Có | +23.757% | +15.342% | +11.587% |
| 20 | Có | +26.784% | +14.881% | +8.889% |
| 21 | Có | +17.404% | +9.593% | +11.958% |
| 22 | — | +2.200% | -6.019% | +19.215% |
| 23 | — | +14.918% | +6.374% | +33.458% |

**fwe**

| Lớp | Thuộc S1 (16–21) | L_ann | L_stab | R_short |
|---|---|---|---|---|
| 0 | — | -20.326% | -53.591% | +3.127% |
| 1 | — | +60.583% | -6.292% | -38.980% |
| 2 | — | +76.924% | +22.110% | -20.915% |
| 3 | — | +78.572% | +31.696% | -10.442% |
| 4 | — | +55.401% | +21.764% | -12.397% |
| 5 | — | +45.132% | +35.323% | -8.745% |
| 6 | — | +42.360% | +23.244% | -2.507% |
| 7 | — | +16.567% | +15.455% | +9.365% |
| 8 | — | +6.527% | -15.186% | -5.678% |
| 9 | — | +7.116% | -0.864% | +8.000% |
| 10 | — | +8.031% | +3.221% | +4.188% |
| 11 | — | +18.506% | +96.741% | +19.299% |
| 12 | — | +60.963% | +85.823% | +9.300% |
| 13 | — | +2.710% | +2.083% | +3.189% |
| 14 | — | +8.401% | +7.849% | +12.745% |
| 15 | — | +11.982% | +11.324% | +11.814% |
| 16 | Có | +12.805% | +12.232% | +12.341% |
| 17 | Có | +21.263% | +14.987% | +12.186% |
| 18 | Có | +20.806% | +15.162% | +11.832% |
| 19 | Có | +26.496% | +16.577% | +11.900% |
| 20 | Có | +26.884% | +14.817% | +6.923% |
| 21 | Có | +16.451% | +14.298% | +11.929% |
| 22 | — | +3.518% | -6.876% | +12.520% |
| 23 | — | +18.409% | +8.628% | +29.258% |

**nem**

| Lớp | Thuộc S1 (16–21) | L_ann | L_stab | R_short |
|---|---|---|---|---|
| 0 | — | +13.859% | +21.883% | +28.365% |
| 1 | — | +289.976% | +58.953% | -36.794% |
| 2 | — | +113.437% | +46.852% | -27.717% |
| 3 | — | +53.869% | +26.070% | -8.721% |
| 4 | — | +31.261% | +18.413% | -7.027% |
| 5 | — | +15.415% | +3.317% | -3.990% |
| 6 | — | +19.794% | +3.769% | -3.328% |
| 7 | — | +21.647% | +44.157% | +5.079% |
| 8 | — | +32.240% | +66.522% | +9.085% |
| 9 | — | +20.350% | +8.117% | +8.746% |
| 10 | — | +3.333% | -5.320% | +2.578% |
| 11 | — | +4.017% | +5.575% | +8.508% |
| 12 | — | +4.458% | -0.303% | +7.926% |
| 13 | — | +3.889% | -0.451% | +9.045% |
| 14 | — | +14.982% | +1.607% | +2.809% |
| 15 | — | +7.495% | -1.922% | +6.753% |
| 16 | Có | +9.164% | +4.155% | +7.401% |
| 17 | Có | +15.962% | +8.971% | +11.106% |
| 18 | Có | +18.474% | +10.149% | +10.527% |
| 19 | Có | +21.095% | +10.804% | +9.735% |
| 20 | Có | +12.651% | +9.703% | +10.978% |
| 21 | Có | +26.291% | +52.902% | +5.541% |
| 22 | — | +3.163% | -4.316% | +19.650% |
| 23 | — | +10.282% | +13.278% | +30.979% |

### 5. Đủ 10 bucket của mỗi loại, ở DCLM và FWE

Bucket đánh số 0–9. Frequency 0 → 9 là hiếm → phổ biến; difficulty 0 → 9 là dễ → khó. `g_par = P300 − P1T`; `g_stu = student P300 − student P1T`; `retention = g_stu/g_par`; `A_init = student X72 − student P300`. Cờ mẫu số nhỏ dùng đúng |g_par| < 0.005.

**dclm / frequency**

| Bucket | g_par | g_stu | Retention | A_init | Cờ |
|---|---|---|---|---|---|
| 0 | 0.099120 | 0.057962 | 0.584768 | 0.024339 | — |
| 1 | 0.083692 | 0.048624 | 0.580988 | 0.019582 | — |
| 2 | 0.073324 | 0.042312 | 0.577049 | 0.017451 | — |
| 3 | 0.063349 | 0.035872 | 0.566261 | 0.015742 | — |
| 4 | 0.054475 | 0.031209 | 0.572901 | 0.014614 | — |
| 5 | 0.048625 | 0.028305 | 0.582100 | 0.010856 | — |
| 6 | 0.025607 | 0.014761 | 0.576457 | 0.006519 | — |
| 7 | 0.027976 | 0.017081 | 0.610563 | 0.006122 | — |
| 8 | 0.018037 | 0.009335 | 0.517540 | 0.002467 | — |
| 9 | 0.014635 | 0.006671 | 0.455800 | 0.003439 | — |

**dclm / difficulty**

| Bucket | g_par | g_stu | Retention | A_init | Cờ |
|---|---|---|---|---|---|
| 0 | 0.001850 | 0.001616 | 0.873534 | 0.000424 | \|g_par\| < 0.005 |
| 1 | 0.009364 | 0.006933 | 0.740349 | 0.004126 | — |
| 2 | 0.019733 | 0.013500 | 0.684119 | 0.007281 | — |
| 3 | 0.029799 | 0.019845 | 0.665960 | 0.009830 | — |
| 4 | 0.042132 | 0.027615 | 0.655456 | 0.012284 | — |
| 5 | 0.057056 | 0.035906 | 0.629315 | 0.014434 | — |
| 6 | 0.072586 | 0.043404 | 0.597959 | 0.016524 | — |
| 7 | 0.090380 | 0.052377 | 0.579519 | 0.018102 | — |
| 8 | 0.104090 | 0.056305 | 0.540925 | 0.018760 | — |
| 9 | 0.081852 | 0.034632 | 0.423107 | 0.019368 | — |

**fwe / frequency**

| Bucket | g_par | g_stu | Retention | A_init | Cờ |
|---|---|---|---|---|---|
| 0 | 0.098981 | 0.058199 | 0.587979 | 0.021286 | — |
| 1 | 0.082634 | 0.047321 | 0.572655 | 0.020242 | — |
| 2 | 0.073946 | 0.039976 | 0.540609 | 0.018635 | — |
| 3 | 0.062833 | 0.034692 | 0.552134 | 0.014979 | — |
| 4 | 0.055534 | 0.028920 | 0.520766 | 0.017647 | — |
| 5 | 0.046672 | 0.024871 | 0.532886 | 0.010718 | — |
| 6 | 0.030062 | 0.015373 | 0.511378 | 0.006779 | — |
| 7 | 0.026566 | 0.015921 | 0.599292 | 0.006874 | — |
| 8 | 0.027635 | 0.011642 | 0.421290 | 0.009102 | — |
| 9 | 0.014376 | 0.007713 | 0.536530 | 0.003674 | — |

**fwe / difficulty**

| Bucket | g_par | g_stu | Retention | A_init | Cờ |
|---|---|---|---|---|---|
| 0 | 0.002011 | 0.000329 | 0.163676 | 0.002583 | \|g_par\| < 0.005 |
| 1 | 0.011256 | 0.007530 | 0.669029 | 0.004696 | — |
| 2 | 0.022889 | 0.015038 | 0.656991 | 0.007592 | — |
| 3 | 0.034231 | 0.022213 | 0.648912 | 0.010048 | — |
| 4 | 0.046743 | 0.030067 | 0.643235 | 0.012086 | — |
| 5 | 0.058689 | 0.036433 | 0.620777 | 0.014874 | — |
| 6 | 0.073518 | 0.043241 | 0.588178 | 0.017019 | — |
| 7 | 0.090044 | 0.049412 | 0.548751 | 0.018384 | — |
| 8 | 0.101537 | 0.048872 | 0.481323 | 0.021593 | — |
| 9 | 0.078322 | 0.031493 | 0.402096 | 0.021062 | — |

Phụ lục này không thay đổi decision đã công bố. Nguồn: [decision_mech.json](decision_mech.json), [noise_paired.json](noise_paired.json), [layers.json](layers.json), [buckets.json](buckets.json) và A2 Q `decision_quant.json` đã khóa SHA trong config.

<!-- A2_M_ADDENDUM_END -->
