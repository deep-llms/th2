# Stage1 — kết quả cuối, 30/10/2026 (SGT)

Trạng thái: **HOÀN TẤT, đã kiểm tra độc lập training, downstream và burn handoff.**

Theo [spec v3.3](A2_stage1_spec.md), trên CulturaX final: **E(3.2B)=+0.021670885 nats → “persists”**; **A_init(3.2B)=+0.012113132 nats → “anneal first”**. P1T chịu mức tăng NLL lớn hơn so với parent riêng, dù NLL tuyệt đối của student P1T vẫn thấp hơn P300. Đây là kết luận của recipe và seed đã đăng ký; X72–P300 không tách riêng tác động LR state.

Downstream bổ sung chưa xác lập khác biệt cho hai tương phản chính: các CI95% của P1T−P300 drop và X72−P300 accuracy đều chứa 0, cả MC mean lẫn LAMBADA. Không đổi nhãn NLL bằng kết quả downstream; không khẳng định hai phương án tương đương.

## Recipe, dữ liệu và kiểm tra

15/15 fits hoàn thành: 13 fits đăng ký và đúng một edge pair LR=3e-3. S1 bỏ original layers 16–21, giữ [0..15,22,23], từ 24 xuống 18 layers. P300/P1T dùng parent và teacher tương ứng; X72 khởi tạo S72 (iter_0072000), teacher P300. Mỗi fit bắt đầu mới với optimizer và lịch LR đầy đủ; student FP32 master/BF16 AMP, teacher BF16, pure full-vocabulary KL. 8 GPU, microbatch4, chunk256, global64×4096 tokens/update.

DCLM được shuffle đúng một lần ở cấp document, NumPy default_rng(2030), rồi pack và lấy các prefix lồng nhau theo thứ tự. Pool shuffle 29,999,992,832 tokens; mỗi fit chỉ dùng ngân sách .2/.8/3.2B, không phải 30B. EOS=0; loại document cuối bị cắt và phần dư packing, lệch một row 4096 tokens so với pool gốc.

Hai corpora DCLM/FineWeb-Edu 30B đã qua full hash, native loader, nguồn/tokenizer/prefix và exit0 độc lập. FineWeb-Edu chưa dùng để healing Stage1. Metadata và receipts của snapshot 3.2B cũ được giữ; hai datafiles cũ đã được thay thế và xoá trước heals.

Preflight đã gửi trước launch: T1/T3/T5/T6/T7/T12/T16/T18 PASS; T7 không trùng document id/content hash; T8=0.948931813 s/update; T17 chênh +0.0000127656 nats (<0.002). T18 lưu sequence count/EOS/first-last decode; cả ba parent có DCLM loss thấp hơn CulturaX. [Báo cáo preflight](A2_STAGE1_PREFLIGHT_20261029.md). Toàn bộ fits có full LR schedule, exact 8-rank data coverage, saved/reload roundtrip difference=0, không divergence hoặc rerun.

## Parent và pruning trước healing

| Parent | Cx tuning | Cx final | DCLM held-out | Cx pre | Δpre Cx | DCLM pre | Δpre DCLM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P300 | 2.722719408 | 2.737793032 | 2.506344141 | 3.763775144 | 1.025982112 | 3.555026045 | 1.048681904 |
| P1T | 2.671354022 | 2.686908992 | 2.452078286 | 4.004721485 | 1.317812493 | 3.744879166 | 1.292800880 |
| S72 | 2.896920584 | 2.908575296 | 2.677280175 | 3.854544222 | 0.945968926 | 3.618580658 | 0.941300482 |

E_pre=0.291830382; parent P1T−P300=-0.050884040; annealing gain S72−P300=0.170782264 nats.

## Grid và các fits

| LR | P300 tuning | P1T tuning | Mean tuning |
| --- | --- | --- | --- |
| 0.0001 | 3.044145819 | 3.012119551 | 3.028132685 |
| 0.0003 | 2.975577966 | 2.939432267 | 2.957505117 |
| 0.001 | 2.930716072 | 2.901834545 | 2.916275309 |
| 0.003 | 2.959859914 | 2.959963242 | 2.959911578 |

Chọn common LR=1e-3; cả hai individual best cũng 1e-3. Base grid có best ở upper edge nên chạy đúng edge pair 3e-3; không mở rộng tiếp. E(.2B) own-best = common-LR = +0.021096293 nats (DCLM diagnostic +0.025535263).

| Arm | Budget | LR | Steps | Cx final | Cx tuning | DCLM held-out | Δpost Cx | Train seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P300 | 0.2B | 0.0001 | 762 | 3.053886996 | 3.044145819 | 2.820801636 | 0.316093964 | 735.16 |
| P1T | 0.2B | 0.0001 | 762 | 3.021207767 | 3.012119551 | 2.787754776 | 0.334298775 | 732.93 |
| P300 | 0.2B | 0.0003 | 762 | 2.986532984 | 2.975577966 | 2.752503952 | 0.248739952 | 733.02 |
| P1T | 0.2B | 0.0003 | 762 | 2.949752725 | 2.939432267 | 2.715240366 | 0.262843733 | 733.10 |
| P300 | 0.2B | 0.001 | 762 | 2.942554079 | 2.930716072 | 2.709119350 | 0.204761046 | 733.15 |
| P1T | 0.2B | 0.001 | 762 | 2.912766332 | 2.901834545 | 2.680388758 | 0.225857340 | 733.28 |
| P300 | 0.2B | 0.003 | 762 | 2.971008611 | 2.959859914 | 2.744826744 | 0.233215579 | 732.75 |
| P1T | 0.2B | 0.003 | 762 | 2.969984805 | 2.959963242 | 2.745557866 | 0.283075813 | 732.81 |
| P300 | 3.2B | 0.001 | 12207 | 2.839891311 | 2.827242626 | 2.611175837 | 0.102098279 | 11721.31 |
| P1T | 3.2B | 0.001 | 12207 | 2.810678156 | 2.798194436 | 2.579649788 | 0.123769164 | 11721.75 |
| X72 | 3.2B | 0.001 | 12207 | 2.852004443 | 2.840085142 | 2.624061857 | -0.056570853 | 11722.37 |
| P300 | 0.8B | 0.001 | 3051 | 2.879720978 | 2.867910781 | 2.649395649 | 0.141927946 | 2930.81 |
| P1T | 0.8B | 0.001 | 3051 | 2.849623360 | 2.838170950 | 2.617139876 | 0.162714369 | 2931.86 |
| X72 | 0.8B | 0.001 | 3051 | 2.918631007 | 2.907669195 | 2.689506000 | 0.010055711 | 2932.38 |
| X72 | 0.2B | 0.001 | 762 | 3.015278782 | 3.005272620 | 2.783838155 | 0.106703486 | 732.86 |

## Quyết định và xu hướng

| Budget | E CulturaX | A_init CulturaX | E DCLM diagnostic | A_init DCLM diagnostic |
| --- | --- | --- | --- | --- |
| 0.2B | 0.021096293 | 0.072724704 | 0.025535263 | 0.074718806 |
| 0.8B | 0.020786423 | 0.038910029 | 0.022010082 | 0.040110351 |
| 3.2B | 0.021670885 | 0.012113132 | 0.022739805 | 0.012886020 |

E(3.2)/E(.8)=1.042549987: ổn định theo heuristic ≥0.5. A_init ratio=0.311311305: vẫn đang thu hẹp (<0.5). Đây là mô tả xu hướng, không ngoại suy sang 12.8B.

| Decision | INCOMPLETE | Positive clause | Negative clause | Small-effect clause | grey |
| --- | --- | --- | --- | --- | --- |
| Length | false | persists (E≥.008): true | reversed (E≤−.008): false | erased (abs(E)<.003): false | false |
| State | false | anneal first (A≥.01): true | prune before you anneal (A≤−.01): false | no effect (abs(A)<.003): false | false |

Phase0A T14 noise: identical rerun −0.000006436, alternate order −0.002285655 nats; đặt cạnh E3.2=+0.021670885 và A3.2=+0.012113132. Đây là hai kiểm tra đơn lẻ, không phải CI hoặc ước lượng phương sai seed.

**Điều kiện healing CulturaX lịch sử:** Phase0A DCLM/S1 có E(.05B)=+0.019603985 và E(.2B)=+0.010876774, LR3e-4. Stage1 DCLM healing có E(.2B)=+0.021096293, LR1e-3. Cùng .2B nhưng khác healing data và cách chọn LR; không diễn giải như ablation riêng corpus. DCLM-held-out cho các fits lịch sử không có trong receipt so sánh này. [Số liệu lịch sử](results/a2_stage1_20261029/historical_comparison.json).

**DCLM held-out — own corpus, chỉ diagnostic:** E_pre=0.244118976, parent P1T−P300=-0.054265854, annealing gain=0.170936035, E ratio=1.033154032, A ratio=0.321264202. Các nhãn quyết định không dùng tập này. P1T có thể đã đọc khoảng 25% corpus so với khoảng 7.5% của P300; mức exposure khác nhau là một phần câu hỏi thực tiễn. CulturaX final trung tính tránh tính lợi thế exposure lần thứ hai trong đánh giá.

## Downstream và paired CI

Sáu model cố định (ba parent và ba student 3.2B), cùng 78,259 requests/model, 9 tasks, zero-shot, seed42, context4096, batch2. Tám task MC lấy mean trọng số bằng nhau; LAMBADA báo riêng. Điểm dưới đây là %, khoảng tin cậy và drop là điểm phần trăm (pp). Drop=parent−student; X72−P300 accuracy dương nghĩa X72 tốt hơn.

| Model | MC mean % | LAMBADA % |
| --- | --- | --- |
| P300-parent | 62.4059 | 65.4764 |
| P1T-parent | 64.4467 | 67.6499 |
| S72-parent | 59.1902 | 57.0541 |
| P300-student | 58.9339 | 61.0712 |
| P1T-student | 61.1314 | 62.3520 |
| X72-student | 58.5309 | 60.8966 |

| Contrast | MC mean pp [95% CI] | LAMBADA pp [95% CI] |
| --- | --- | --- |
| P300_drop | +3.4720 [+2.8848, +4.0657] | +4.4052 [+3.4737, +5.3367] |
| P1T_drop | +3.3153 [+2.7437, +3.8887] | +5.2979 [+4.3470, +6.2682] |
| X72_drop_from_S72 | +0.6593 [-0.0003, +1.3186] | -3.8424 [-4.9680, -2.7363] |
| P1T_minus_P300_drop | -0.1567 [-0.9051, +0.5989] | +0.8927 [-0.3493, +2.1153] |
| X72_minus_P300_accuracy | -0.4030 [-0.8798, +0.0723] | -0.1747 [-0.9509, +0.6016] |
| X72_minus_P300_drop | -2.8127 [-3.5981, -2.0496] | -8.2476 [-9.5672, -6.9668] |

CI dùng paired bootstrap 10,000 resamples, seed0, cùng item giữa các model; multinomial counts của các paired rows giống nhau tương đương resampling items. CI phản ánh benchmark items, không phản ánh training seeds. Các CI per-task được giữ đầy đủ trong paired_accuracy.json; không dùng các task chọn riêng để thay kết luận tổng.

| Task | P300-parent | P1T-parent | S72-parent | P300-student | P1T-student | X72-student |
| --- | --- | --- | --- | --- | --- | --- |
| hellaswag | 65.057 | 68.592 | 58.076 | 59.391 | 61.840 | 58.514 |
| arc_easy | 67.003 | 68.434 | 61.237 | 63.047 | 64.478 | 62.374 |
| arc_challenge | 37.116 | 40.358 | 34.130 | 33.618 | 34.727 | 32.253 |
| piqa | 75.680 | 76.551 | 72.416 | 73.721 | 73.993 | 73.341 |
| winogrande | 62.036 | 63.536 | 59.274 | 60.379 | 61.168 | 59.511 |
| boolq | 62.355 | 65.902 | 63.089 | 55.015 | 62.844 | 56.055 |
| openbookqa | 39.200 | 39.200 | 36.800 | 35.600 | 37.200 | 36.400 |
| sciq | 90.800 | 93.000 | 88.500 | 90.700 | 92.800 | 89.800 |
| lambada_openai | 65.476 | 67.650 | 57.054 | 61.071 | 62.352 | 60.897 |

Hai actual GPU gates P300 parent/student PASS: manual loglikelihood gap 1.32685e-7; batch/production differences 0 và 0.000126839; greedy mismatch=0. Scores dùng FP32 parameters/BF16 AMP, exact six identities. Canonical CPU replay dùng đúng lm-eval0.4.13 settings/metrics; warning git metadata ngoài repository không làm child fail. [Kết quả đầy đủ](results/a2_stage1_20261029/downstream_final/supervised/run/downstream/results.json), [CI đầy đủ](results/a2_stage1_20261029/downstream_final/supervised/run/downstream/paired_accuracy.json).

Kiểm tra CPU độc lập hoàn tất 30Oct00:54:01 SGT: kiểm hash toàn bộ score/replay, khớp item/protocol/identities, tái lập aggregate và paired bootstrap 10,000 seed0 khớp chính xác. [Receipt độc lập](results/a2_stage1_20261029/downstream_final/independent_replay_verification.json).

## Compute, handoff và lưu trữ

Tổng 52,632 updates = 13,797,163,008 tokens qua 15 fits; đây là tổng exposure qua các fits, không phải số token duy nhất. Training coordinator 51,971.41 s (14.44 node-hours); ledger preflight 1,800 s. Downstream supervisor từ 00:00:30 đến 00:26:16: 1546.51 s. Ledger bảo thủ gồm khoảng chuyển tiếp, preparation và downstream: 56003.94 s = **15.5567 node-hours**, dưới report30h/hard36h; tương đương 124.45 GPU-hours nếu tính cả node8GPU, không phải active kernel time. Corpus preparation chạy CPU riêng và không tính vào GPU ledger.

Tính bảo thủ cả thời gian chờ receipt và hoàn thiện báo cáo đến 30Oct00:57:12 SGT: 16.0899 node-hours kể cả ledger preflight, vẫn dưới 30/36 giờ.

Training terminal 29Oct23:48:01; independent burn handoff23:52:24. Downstream terminal30Oct00:26:16; independent snapshot00:47:30 xác nhận supervisor exit0, guard released, session ended, 8 burn workers710843–710850; collective cycles1450→1460, payload1610.01→1621.11GiB. Không dùng việc PID biến mất làm bằng chứng thành công.

Nguồn training0c1c148 (103 files), downstream13686ed (105 files); numerical core đóng băng. Spec SHA `9ebe0d0a377f0107f7058bf5e4899b5380cbe5f4f518b09091ac533d2c9f254e`. Resource locks: [terminal](../resources/a2_stage1_terminal_20261029.json), [reference](../resources/a2_stage1_reference_20261029.json). [Training archive](results/a2_stage1_20261029/training_final/verification.json), [downstream terminal verification](results/a2_stage1_20261029/downstream_final/verification.json), [final decision](results/a2_stage1_20261029/final/decision_stage1.json).

Artifacts gốc được giữ trên node: training `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-heals-20261029-a01/supervised/run/results/a2_stage1/`; downstream `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-downstream-20261029-a01/supervised/run/downstream/`. Per-item scores/replays, BF16 healed checkpoints, configs/manifests và nguyên bản training decision còn nguyên. Để giữ native supervisor, downstream nằm ở root riêng thay vì di chuyển vào training root; archive cuối liên kết hai phần và bổ sung final decision, không sửa provenance gốc. JSON summaries local bỏ các mảng per-sequence lớn; raw receipts lưu local với SHA, không đưa vào Git.

Phạm vi Stage1 kết thúc ở báo cáo này. 12.8B, FineWeb-Edu healing và replication seed cần quyết định tiếp theo của chủ dự án; chưa khởi chạy.
