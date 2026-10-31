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

Toàn bộ bảng và per-sequence files có ở máy này, giữ ngoài Git để đáp ứng giới hạn code sync 25 MiB. Báo cáo, receipt và script tái kiểm chứng được lưu trong Git. **18 per-token arrays (~726 MB) giữ trên B200**, đã được hash và replay tại node; chưa sao lưu chúng sang máy phát triển. Vị trí từng file có trong token_inventory.json và manifest.

Không còn bước đã đăng ký chờ thực hiện. Cờ còn giữ: lỗi điều phối gốc, bucket mẫu số nhỏ và các hạn chế diễn giải nêu trên.
