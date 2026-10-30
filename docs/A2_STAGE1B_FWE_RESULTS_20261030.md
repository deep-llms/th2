# Stage1b FineWeb-Edu — kết quả các runs đợt 30/10/2026

File kết quả chung cho đợt **Stage1b FineWeb-Edu bắt đầu ngày 30/10/2026**, gồm Part1, Part2 và downstream liên quan. Đặt cùng nhóm file kết quả A2; [kết quả Stage1 DCLM](A2_STAGE1_RESULTS_20261030.md) là đối chiếu của đợt trước. Artifact của đợt FWE nằm trong `docs/results/a2_stage1b_fwe_20261030/`.

Cập nhật báo cáo: **31/10/2026 02:16 SGT**. Bằng chứng khoa học mới nhất là phép tính lại độc lập lúc **31/10/2026 00:39:22 SGT**, được nhận lúc 01:00 SGT. Tất cả thời gian bên dưới dùng SGT.

**Training đã hoàn tất và được xác minh đủ 15/15 fit. Downstream sáu model chưa hoàn tất.** Kết quả NLL giữ nhãn `persists`, `replicated`, `anneal first`. Downstream đã vượt qua sáu gate đại diện nhưng dừng vì một request trong phép so sánh P300 toàn bộ vượt ngưỡng LL đang được mã thực thi áp dụng. Lỗi đã được tái hiện độc lập; chưa có kết quả năm model còn lại hoặc paired CI.

## Kết luận đã xác minh

| Hạng mục | Kết quả |
|---|---|
| LR chung trên FWE | **1e-3**; LR tốt nhất riêng của P300 và P1T cũng là 1e-3 |
| E ở 3.2B | **+0.023461150403907727 nats**, nhãn `persists` |
| Đối chiếu hiệu ứng E với DCLM | `replicated` theo quy tắc đã đăng ký |
| A_init ở 3.2B | **+0.01299365118963669 nats**, nhãn `anneal first` |
| A_init 3.2B / 0.8B | **0.30606568523151645**, còn khoảng 30.61% mức ở 0.8B |
| Training | 15 fit hoàn tất; full LR schedule, exact 8-rank order, save/reload sai khác 0, không divergence |
| DCLM P300 regression | 9 task accuracy khớp chính xác Stage1; LL max difference = 0 trên 78,259 requests |
| Sáu gate FWE đại diện theo ngưỡng sửa đổi | Cả sáu PASS; mỗi model 0/502 token mismatch |
| P300 full batch1 và batch2 | Cả hai đủ 78,259 scores; tiêu chí MC mean/LAMBADA đạt |
| Kiểm tra full LL trong mã thực thi | **FAIL**, max 0.10355663299560547 > 0.1 nats/token; 1 request vượt ngưỡng |
| Hoàn tất toàn Stage1b | **Chưa hoàn tất**; các bước còn mở nằm cuối file |

## Thiết lập và cách đọc kết quả

Thực nghiệm theo Stage1b spec v1.1, dùng layer set S1, heal trên FineWeb-Edu, document shuffle một lần bằng seed 2030, pack 4096 và các prefix lồng nhau. P300 và P1T là hai parent của đối chiếu training length; X72 là student từ S72 trong đối chiếu initialization/annealing.

NLL càng thấp càng tốt. LR được chọn bằng trung bình **CulturaX tuning NLL** của P300 và P1T. **CulturaX final NLL** là endpoint quyết định; FWE held-out chỉ dùng chẩn đoán.

- Δ_post = NLL student − NLL parent tương ứng.
- E = Δ_post(P1T) − Δ_post(P300). E dương nghĩa là P1T chịu mức tăng NLL lớn hơn P300 so với parent tương ứng, dù NLL tuyệt đối của student P1T có thể thấp hơn.
- A_init = NLL(X72 student) − NLL(P300 student). A_init dương nghĩa là P300 student có NLL thấp hơn X72 tại cùng budget.
- Các nhãn là kết luận của recipe và ngưỡng đã đăng ký; chúng không tự chứng minh quan hệ nhân quả riêng của LR state hay exposure.

## Chọn LR và toàn bộ 15 fit

Grid ban đầu gồm ba LR 1e-4, 3e-4, 1e-3 trên hai arm tại 0.2B. Vì optimum nằm ở biên trên, đã chạy thêm đúng một cặp 3e-3. Tổng cộng **8 fit grid + 7 fit còn lại = 15 fit**. Không mở rộng LR thêm.

| LR | Trung bình tuning NLL của P300 và P1T tại 0.2B |
|---|---:|
| 0.0001 | 3.138225751921 |
| 0.0003 | 3.068906872385 |
| 0.001 | 3.024409690999 |
| 0.003 | 3.052233397952 |

LR 1e-3 thắng cả lựa chọn chung và lựa chọn riêng từng arm. Lựa chọn trên FWE trùng DCLM nhưng được tính lại độc lập.

| Arm | Budget token | LR | CulturaX tuning NLL | CulturaX final NLL | FWE held-out NLL | Δ_post CulturaX |
|---|---|---:|---:|---:|---:|---:|
| P300 | 0.2B | 0.0001 | 3.151107981 | 3.169482734 | 2.624746937 | +0.308851408 |
| P1T | 0.2B | 0.0001 | 3.125343523 | 3.142402422 | 2.597472721 | +0.333695089 |
| P300 | 0.2B | 0.0003 | 3.084298514 | 3.103321913 | 2.553008765 | +0.242690587 |
| P1T | 0.2B | 0.0003 | 3.053515231 | 3.071785913 | 2.521224369 | +0.263078580 |
| P300 | 0.2B | 0.001 | 3.037060038 | 3.056353314 | 2.505082425 | +0.195721988 |
| P1T | 0.2B | 0.001 | 3.011759344 | 3.030653544 | 2.478802482 | +0.221946210 |
| P300 | 0.2B | 0.003 | 3.052761722 | 3.071304613 | 2.520658197 | +0.210673287 |
| P1T | 0.2B | 0.003 | 3.051705074 | 3.069546812 | 2.518718774 | +0.260839479 |
| P300 | 3.2B | 0.001 | 2.936589511 | 2.955812353 | 2.406616605 | +0.095181027 |
| P1T | 3.2B | 0.001 | 2.908373912 | 2.927349511 | 2.376905284 | +0.118642177 |
| X72 | 3.2B | 0.001 | 2.949474459 | 2.968806004 | 2.420236999 | -0.073345998 |
| P300 | 0.8B | 0.001 | 2.977784103 | 2.997096324 | 2.444843807 | +0.136464998 |
| P1T | 0.8B | 0.001 | 2.949299475 | 2.967992111 | 2.416288128 | +0.159284777 |
| X72 | 0.8B | 0.001 | 3.020255002 | 3.039550123 | 2.488052950 | -0.002601879 |
| X72 | 0.2B | 0.001 | 3.116279572 | 3.134831688 | 2.583314970 | +0.092679686 |

Các giá trị trong bảng được làm tròn; receipt và artifact gốc giữ đầy đủ độ chính xác. Training Part1 kết thúc **30/10 12:43:17**, Part2 kết thúc **30/10 23:21:56**. Native terminal và burn handoff của cả hai phần đã được kiểm tra độc lập.

## Đối chiếu DCLM và FWE trên CulturaX final

Cả hai dataset chọn LR chung và own-best của P300/P1T là 1e-3. Hiệu dưới đây là FWE trừ DCLM, đơn vị nats.

| Budget | DCLM E | FWE E | Hiệu E | DCLM A_init | FWE A_init | Hiệu A_init |
|---|---:|---:|---:|---:|---:|---:|
| 0.2B | 0.021096293348 | 0.026224221566 | +0.005127928219 | 0.072724703707 | 0.078478373608 | +0.005753669902 |
| 0.8B | 0.020786422564 | 0.022819779411 | +0.002033356848 | 0.038910028864 | 0.042453799353 | +0.003543770489 |
| 3.2B | 0.021670884570 | 0.023461150404 | +0.001790265834 | 0.012113131874 | 0.012993651190 | +0.000880519316 |

E vẫn dương ở 3.2B và gần mức 0.8B trên cả hai dataset. A_init giảm theo budget nhưng vẫn dương ở 3.2B.

| Dataset | E 3.2B / 0.8B | A_init 0.8B / 0.2B | A_init 3.2B / 0.8B |
|---|---:|---:|---:|
| DCLM | 1.042549986824 | 0.535031796356 | 0.311311305279 |
| FWE | 1.028105924298 | 0.540961763109 | 0.306065685232 |

## FWE held-out và giới hạn suy luận

| Budget | E trên FWE held-out | A_init trên FWE held-out |
|---|---:|---:|
| 0.2B | 0.026352782881 | 0.078232544929 |
| 0.8B | 0.024077046610 | 0.043209142918 |
| 3.2B | 0.022921405400 | 0.013620394356 |

Own-corpus held-out có thể đã xuất hiện trong pretraining của parent. Exposure ước tính trên FWE là khoảng 71% đối với P1T và 21% đối với P300; trên DCLM khoảng 25% và 7.5%. Vì vậy chỉ số held-out này không dùng để ra quyết định hoặc cô lập tác động exposure. Đây là các recipe một seed, chưa phải CI qua nhiều training seed.

Điều kiện CulturaX lịch sử: Phase0A DCLM/S1 có E(0.05B) ≈ 0.019603985, E(0.2B) ≈ 0.010876774 tại LR 3e-4; Stage1 DCLM có E(0.2B) ≈ 0.021096293 tại LR 1e-3. Khác biệt recipe/LR cần được giữ khi diễn giải, không coi đây là ablation riêng corpus.

## Dữ liệu và kiểm tra trước training

Heal stream là `HuggingFaceFW/fineweb-edu`, prepared revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Release gốc `v1.0.0` ở revision `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`: cả **60 file được chọn có cùng byte/LFS identity**, xác nhận release tương ứng với parent. Corpus shuffle có **29,999,992,832 tokens / 7,324,217 sequences**. Corpus gốc được giữ nguyên.

| Kiểm tra | Kết quả đã xác minh |
|---|---|
| T1 | P300/P1T/S72 PASS; precision gaps lần lượt 0.0000913465 / 0.0001567571 / 0.0000776934 nats |
| T3 | PASS; S72/X72 đủ điều kiện |
| T7 | Document-ID overlap = 0, content-hash overlap = 0 |
| T18 | EOS 29,155,864; share 0.000971862366; không flag |
| T17 | Rerun trước science đã PASS; delta −0.000144556073 trong ±0.002 |

| Parent | CulturaX final NLL | FWE held-out NLL |
|---|---:|---:|
| P300 | 2.860631325857 | 2.308842448625 |
| P1T | 2.808707333602 | 2.256209722936 |
| S72 | 3.042152001946 | 2.489572923114 |

P300/P1T CulturaX final khớp chính xác Phase0A. S72−P300 annealing gain là **0.18152067608948386 nats**. T14a có identical-rerun noise −0.000006436 và alternate-order difference −0.002285655; đây là hai phép đo đơn lẻ, không phải ước lượng variance hay CI nhiều seed.

## Downstream đã thực hiện

Dùng cùng **78,259 canonical requests**, 9 task, zero-shot seed 42, FP32 parameters và BF16 AMP. Sáu model dự kiến là P300/P1T/S72 parents và P300/P1T/X72 students ở 3.2B. Paired bootstrap dự kiến 10,000 resamples, seed 0; hiện chưa có paired CI FWE.

DCLM P300 regression đã PASS với accuracy của cả 9 task giống hệt Stage1 và LL max difference bằng 0. Đây là checkpoint DCLM, không thay thế việc kiểm tra checkpoint FWE.

Lần FWE downstream đầu tiên thất bại trước scoring tại ngưỡng batch 0.01: P300 parent có sai lệch **0.011164188385009766 nats/token**. Sau đó người dùng cho phép ngưỡng gate sửa đổi: manual ≤ 2e-5, batch ≤ 0.1 nats/token, greedy token mismatch ≤ 0.1%; ngưỡng cũ 0.01 giữ làm diagnostic.

### Sáu gate đại diện trên GPU

| Model | Manual nats/token | Batch nats/token | Token mismatch | Gate sửa đổi | Diagnostic ngưỡng cũ 0.01 |
|---|---:|---:|---:|---|---|
| P300-parent | 0 | 0.011164188385 | 0/502 | PASS | FAIL |
| P1T-parent | 2.65370244565e-07 | 0.0292415618896 | 0/502 | PASS | FAIL |
| S72-parent | 0 | 0.00355625152588 | 0/502 | PASS | PASS |
| P300-student | 0 | 0 | 0/502 | PASS | PASS |
| P1T-student | 2.64221455628e-07 | 0 | 0/502 | PASS | PASS |
| X72-student | 0 | 0.0159463882446 | 0/502 | PASS | FAIL |

### So sánh P300 trên toàn bộ requests

Cả batch1 và batch2 đã tạo đủ 78,259 scores. Quy tắc accuracy chấp nhận batch2 nếu độ lệch tuyệt đối của trung bình 8 task MC và LAMBADA đều ≤ 0.1 điểm phần trăm. Kết quả đạt quy tắc này: **MC mean +0.004797435043 điểm phần trăm; LAMBADA 0**. Từng task MC được báo cáo riêng, không áp thêm ngưỡng accuracy riêng từng task.

| Task | Accuracy batch1 (%) | Accuracy batch2 (%) | Batch2 − batch1 (điểm %) | Prediction flips |
|---|---:|---:|---:|---:|
| hellaswag | 61.412069 | 61.422027 | +0.009958 | 7 |
| arc_easy | 72.558923 | 72.474747 | -0.084175 | 5 |
| arc_challenge | 42.576792 | 42.662116 | +0.085324 | 2 |
| piqa | 75.136017 | 75.136017 | +0.000000 | 2 |
| winogrande | 60.931334 | 61.089187 | +0.157853 | 2 |
| boolq | 62.385321 | 62.354740 | -0.030581 | 3 |
| openbookqa | 41.800000 | 41.800000 | +0.000000 | 0 |
| sciq | 93.300000 | 93.200000 | -0.100000 | 1 |
| lambada_openai | 51.775665 | 51.775665 | +0.000000 | 1 |

MC flips tính thay đổi choice theo primary metric, kể cả chuyển giữa hai đáp án sai. LAMBADA có **1 thay đổi token prediction theo teacher forcing nhưng 0 correctness flips**; không coi đó là kết quả sinh tự do.

| Diagnostic toàn bộ | Giá trị |
|---|---:|
| Requests có LL khác 0 | 21,102 / 78,259 |
| Max absolute LL difference, chưa chuẩn hóa token | 0.5551643371582031 nats/request |
| Mean absolute LL difference, chưa chuẩn hóa token | 0.011516675116967773 nats/request |
| Max normalized LL difference | **0.10355663299560547 nats/token** |
| Token mismatch | 1,309 / 1,396,626 = **0.0937258793693%**, đạt ngưỡng 0.1% |
| Requests vượt 0.1 nats/token | **1 / 78,259** |
| Requests trong cặp có padding | 318; 46 có LL khác 0 |
| Requests trong cặp không padding | 77,941; 21,056 có LL khác 0 |

Khác biệt LL không chỉ xuất hiện ở cặp có padding. Request có **raw LL difference lớn nhất** không phải request có **normalized LL difference lớn nhất**.

### Lỗi làm downstream dừng

Ngoài sáu gate đại diện, mã thực thi còn áp ngưỡng **0.1 nats/token cho max trên toàn bộ requests**. Kiểm tra này FAIL vì **0.10355663299560547 > 0.1**, dù quy tắc accuracy chấp nhận batch2. Job thoát mã 1 lúc **31/10 00:12:10 SGT**, trước khi chấm năm model còn lại.

Phép tính lại CPU độc lập lúc **00:39:22 SGT** xác minh 132 source files và tái tạo chính xác toàn bộ batch decision, accuracies, token/prediction flips và per-request LL/padding. Request duy nhất vượt ngưỡng có index **71485**, context **53 tokens**, continuation **1 token**, cặp **71484/71485 không padding**; LL batch2−batch1 = **+0.10355663299560547**, không đổi trạng thái greedy exact-match. Điều này chưa chứng minh nguyên nhân số học cụ thể.

Kết quả FAIL được giữ nguyên. Sáu gate PASS không được dùng để ghi đè lỗi full-set; phép xác minh lỗi không phải successful final replay.

## Bàn giao và thời gian sử dụng

| Mốc | Trạng thái đã xác minh |
|---|---|
| Part1 terminal và handoff | Training kết thúc 30/10 12:43:17; handoff xác minh độc lập 13:00:54 |
| Part2 terminal và handoff | Training kết thúc 30/10 23:21:56; supervisor exit 0; burn handoff xác minh 23:31:20 |
| Final downstream lỗi | 31/10 00:12:10, exit 1 |
| Native handoff sau lỗi | 00:13:16 khôi phục burn và giải phóng guard |
| Kiểm tra burn độc lập mới nhất | 00:33:07: 8 workers 730982–730989 dưới launcher 730913; cycles 1370→1380, collective payload 1521.18→1532.29 GiB |

Native ledger sau training là **71,801.2573 giây ≈ 19.9448 node-giờ**. Ledger elapsed bảo thủ, gồm cả khoảng chờ/controller, là **20.8001 node-giờ** tại downstream failure và **21.1493 node-giờ** tại kiểm tra 00:33:07. Các số này không phải tổng GPU compute thuần hoặc số đo sử dụng hiện tại sau mốc đó.

Theo override 30/10 17:30 SGT, các mốc 20/24 node-giờ và deadline cũ chỉ dùng báo cáo. Downstream cuối có timeout/deadline null; lỗi trên không do timeout. Legacy timeout Part2 đã hết hiệu lực khi job hoàn tất. `commands.sh` đang ở `#0`, không receipt chờ và chưa gửi GPU retry. Watcher hiện có tiếp tục với `until=null`; không suy ra trạng thái GPU hiện tại từ receipt burn 00:33:07.

## Provenance và bằng chứng

| Thành phần | Commit thực thi |
|---|---|
| Part1 training | `7dd9da088ba5af581c295775819319ce54b2025c` |
| Part2 training | `42f7820ce72d932ff3d75edc96e16538dbd85df5` |
| Preflight provenance kế thừa | `9f5601d97ddf2940b32ba5e17de8d21b7c7fe1c0` |
| Final downstream | `f0c8154808bef431c76553a47fd9727d3ba8bd53` |

27 numerical core hashes của training khớp provenance kế thừa. Final downstream giữ nguyên 66 file eval/model của regression và thêm 6 adapter đã đăng ký. Published results và artifact gốc được giữ nguyên. Runtime audit ghi nhận software và cùng 8 GPU UUID, nhưng chưa ghi đầy đủ cuDNN/NCCL, driver package và actual SDPA backend; không dùng phần thiếu này để đổi nhãn T17.

Bằng chứng chính có thể truy từ các file sau:

- [Stage1b spec v1.1](A2_stage1b_fwe_spec.md), SHA `b2121a4f2859f99db421273082c6be035dc5e833af6d1d071f5ed4cd381ed483`.
- [Release proof FWE](../resources/a2_stage1b_fwe_release_20261030.json) và [preflight](A2_STAGE1B_PREFLIGHT_20261030.md).
- [Part1 terminal](results/a2_stage1b_fwe_20261030/part1_final/verification.json), [Part2 terminal](results/a2_stage1b_fwe_20261030/part2_final/verification.json), [training decision](results/a2_stage1b_fwe_20261030/part2_final/decision_stage1_fwe.json).
- [DCLM Stage1](A2_STAGE1_RESULTS_20261030.md) và [runtime audit](A2_STAGE1B_FOLLOWUP_REPLY_20261030.md).
- [Sáu gate](results/a2_stage1b_fwe_20261030/downstream_final_0001/receipt.json), [full batch decision](results/a2_stage1b_fwe_20261030/downstream_final_batch_failure/batch_decision.json), [failed-job handoff](results/a2_stage1b_fwe_20261030/downstream_final_batch_failure/receipt.json).
- [Replay độc lập của lỗi](results/a2_stage1b_fwe_20261030/downstream_final_batch_failure/independent_failed_batch_replay.json), raw SHA `aa54ba2393fc93c28aeaca73f0a13c0bd9bf2b0ed7bcdae6341752dd209ddb46`.

P300 score SHA batch1: `231aea3bddb4be446aa949cef479cc0e4224720792f8892ad4b7dbdc82b5d1dd`; batch2: `ae05ee601822e3af5c24b70b1e6ea1c974b07fb9967ed72acc9743953adca3a3`. Canonical prepared requests SHA: `85cf7f2119e35e81a79aea48f58080bdc9ba5371b27accbda0eac52aa6770bbf`.

## Quyết định continuation ngày 31/10/2026

**Option 2 đã được phê duyệt.** Accuracy rule là tiêu chí chọn batch và đã PASS. Full-set LL max 0.10355663299560547 tại request 71485 cùng token mismatch 0.0937258793693% chỉ là diagnostic. Ngưỡng 0.1 nats/token áp dụng cho sáu gate đại diện. Tái dùng scores P300-parent batch2 và chạy năm model còn lại; cập nhật kết quả thực tế sau receipt. Deviation đã ghi trong [REPORT.md](REPORT.md).

## Các vấn đề còn mở

| Vấn đề | Cần giải quyết |
|---|---|
| **Triển khai quyết định batch đã chốt** | Người dùng đã chọn Option 2: full-set LL là diagnostic, batch2 giữ nguyên; continuation được phép launch ngay. |
| **Năm model chưa chấm production** | P1T-parent, S72-parent, P300-student, P1T-student, X72-student. P300-parent đã có cả batch1 và batch2 để tái dùng sau quyết định; chưa có bộ production sáu model hoàn chỉnh. |
| **Paired CI và replay kết quả cuối** | Cần bộ scores/replays sáu model cùng batch, paired bootstrap 10,000 resamples seed 0 và xác minh độc lập. Replay đã có chỉ xác minh lỗi P300. |
| **Bàn giao sau một lượt tiếp tục** | Nếu được tiếp tục, cần fresh PID/start/source/ownership/free-GPU checks trước launch, rồi native terminal và burn progress sau khi hoàn tất. Handoff của job lỗi hiện đã được xác minh. |
| **Chốt toàn Stage1b** | Bổ sung downstream, paired CI, compute ledger cuối và decision/report hoàn tất sau khi các điều kiện trên đạt. Báo cáo này tổng hợp kết quả hiện có, không chứng nhận toàn Stage1b hoàn tất. |

Các hướng từng được đưa ra (lịch sử): **(1)** chuyển cả sáu model sang batch1, tái dùng P300 batch1 và giữ nguyên diagnostic FAIL; **(2)** xác nhận ngưỡng 0.1 chỉ áp dụng cho gate đại diện, giữ full-set LL là diagnostic và tiếp tục batch2 theo accuracy rule; **(3)** giữ downstream dừng để điều tra thêm. **Đã được người dùng chọn hướng (2), yêu cầu launch ngay:** tái dùng P300-parent batch2, chấm năm model còn lại ở batch2 rồi paired bootstrap và independent replay. Kiểm tra khoa học log/flag/continue; kết quả lỗi hoặc thiếu không được ghi thành PASS. Không chạy thêm training, T17, LR hoặc exposure experiment ngoài phạm vi đã đăng ký.
