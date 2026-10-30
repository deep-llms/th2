# Stage1b: trả lời ba follow-up trước interim

Ngày 30/10/2026. Đối chiếu từ log/receipt đã có, **không dùng GPU và không dừng P1T**. Đây là phần bổ sung cho [xác nhận trước](A2_STAGE1B_CONFIRMATIONS_REPLY_20261030.md). Mục 1a–1c và 3 của lần xác nhận trước đã được user chấp nhận; không mở thêm gate T17.

## 1. T17 so với Stage1

| Giá trị | NLL / chênh lệch (nats) |
|---|---:|
| Phase0A original | 3.0618757520419022 |
| Stage1 T17 rerun | 3.0618885176067017 |
| Stage1 − original | +0.0000127655647995 |
| Stage1b T17 rerun | 3.0617311959690636 |
| Stage1b − original | -0.0001445560728386 |
| **Stage1b rerun − Stage1 rerun** | **-0.0001573216376380** |
| T14a identical-rerun difference | -0.0000064359164353 |

Độ lớn chênh lệch giữa hai rerun bằng **24.44×** chênh lệch T14a đã quan sát. Đây là tỷ lệ giữa hai quan sát, không phải một ước lượng phương sai hay kiểm định significance. Không thể dùng “core byte-identical” để khẳng định kết quả bitwise-identical. Log hiện có **chưa xác định nguyên nhân** chênh lệch; không có bằng chứng đủ để quy cho một thay đổi phiên bản hay kernel cụ thể. Hai T17 đều nằm trong ±0.002 của original. Không thay đổi label/threshold 0.003 và 0.008 vì audit này.

### 1a. Runtime Stage1 và Stage1b

| Mục | Stage1 T17 | Stage1b T17 | Bằng chứng / giới hạn |
|---|---|---|---|
| torch | `2.14.1+cu130` | `2.14.1+cu130` | Hai `accelerate-env.log` giống byte; provenance ghi `2.14.1`. |
| CUDA | Build tag `cu130` | Build tag `cu130` | Tag CUDA 13.0 của torch build. Không có bản ghi exact loaded CUDA runtime cho từng T17; không đồng nhất build tag với mọi CUDA library được nạp. |
| cuDNN | Không ghi trong receipt T17 | Không ghi trong receipt T17 | Không có version để xác nhận bằng nhau. |
| Attention interface | Native OpenSci, PyTorch `sdpa` | Native OpenSci, PyTorch `sdpa` | `load_parent`/`assert_native` buộc SDPA, cache off, attention dropout 0; model code SHA giữ nguyên. |
| Attention kernel được chọn thực tế | Không được trace | Không được trace | Không thể xác nhận Flash / memory-efficient / math / cuDNN SDPA backend từ các log này. |
| transformers | `4.57.1` | `4.57.1` | Provenance của cả hai T17. |
| accelerate | `1.13.0` | `1.13.0` | Provenance và environment log. |
| NCCL tại T17 | Không ghi version | Không ghi version | Không khẳng định equality của NCCL chỉ từ cùng torch/env path. |
| NVIDIA driver package tại T17 | Không ghi version | Không ghi version | Không có `nvidia-smi driver_version` cho hai T17 trong evidence đã lưu. |
| Python | `3.11.15` | `3.11.15` | Environment log. |
| NumPy / safetensors | `2.4.6` / `0.8.0` | `2.4.6` / `0.8.0` | Provenance. |
| Environment | `a2_env` | `a2_env` | Cùng native interpreter path trong job commands. |
| Host | `thiennh-p6-6y3o-worker-0` | Như Stage1 | GPU inspection. |
| GPUs | 8×NVIDIA B200 | Cùng 8 GPU UUID, cùng ánh xạ index | Đã so khớp cả 8 UUID, không chỉ tên GPU. Không thấy thay node/GPU giữa hai snapshot. |
| Accelerate/precision | 8 process, BF16 mixed precision | Như Stage1 | Student FP32 params, teacher BF16; cùng implementation autocast/score. |

Thông tin lịch sử bổ trợ, **không gán thành runtime chính xác của T17**: `temp/deploy_health01.log` có `NCCL version 2.30.7+cuda13.3`; `temp/a2_stage1_heals_health02` có dòng NET/OFI `CUDA driver version 13000 with runtime 12090`. Dòng thứ hai là thông tin CUDA API/runtime của plugin, **không phải số phiên bản NVIDIA driver package**. Không có dòng tương ứng đủ để chứng minh chúng giống nhau ở cả hai T17.

### 1b. T17 có đi qua bốn file sửa không?

| File sửa | Vai trò với T17 |
|---|---|
| `eval/a2_checkpoint.py` | Không nằm trong đường T17 training/scoring. Đây là checkpoint identity của downstream. |
| `eval/a2_run.py` | Không; T17 không gọi coordinator downstream. |
| `scripts/a2_stage1_data.py` | Không dùng để tạo stream T17. T17 đọc lại CulturaX Phase0A gốc, không đọc FWE shuffle. File này phục vụ preparation của stream FWE. |
| `scripts/launch_a2_downstream.py` | **Có ở bước chuẩn bị:** launcher preflight import/call `freeze_source`. Thay đổi là thêm tham số `extra_files=()`; lời gọi T17/preflight dùng mặc định. Nó copy source, không tính forward/loss/update/val. |

Đường chạy số của T17 là `efficient_llm.a2_stage1b.model_stage` → `efficient_llm.a2.train.heal` / `a2.model` / `a2.measure`. Adapter mới có nhánh T17 tương ứng nhánh Stage1 cũ, dùng `nem/P300/S1`, LR 3e-4, 190 update, cùng CulturaX data SHA, cùng 64×4096 token/update, microbatch 4/chunk 256. Exact 8-rank order và full 190-update LR sequence đã kiểm tra. Cả bốn file sửa trên không thay công thức số được thực thi trong T17.

### 1c. FWE parent final-set val đã đo ở Phase0A

Đã đo. Đây là **CulturaX final-set NLL**, không phải FWE held-out:

| Parent | Phase0A | Stage1b | Stage1b − Phase0A |
|---|---:|---:|---:|
| FWE P300 | 2.8606313258568150 | 2.8606313258568150 | 0.0000000000000000 |
| FWE P1T | 2.8087073336018160 | 2.8087073336018160 | 0.0000000000000000 |

Hai số khớp chính xác ở toàn bộ độ chính xác đã lưu, chặt hơn mốc 1e-5. Điều này là bằng chứng trực tiếp rằng đánh giá hai parent trên final set tái lập được; nó không chứng minh toàn bộ 190-update training là deterministic và không tự giải thích sai khác T17.

Nguồn: [Phase0A summary](results/a2_remaining_phase0a_20261028/final/summary.json), [Stage1b P300](results/a2_stage1b_fwe_20261030/preflight_final/supervised/run/P300.json), [Stage1b P1T](results/a2_stage1b_fwe_20261030/preflight_final/supervised/run/P1T.json). Bản full receipt local được dùng để đối chiếu; SHA ghi trong [evidence](results/a2_stage1b_fwe_20261030/followup_audit/evidence.json).

## 2. Provenance trong interim và final

Sẽ ghi rõ nguyên văn ý sau:

> Các run Stage1b Part1 được thực thi từ frozen source `7dd9da088ba5af581c295775819319ce54b2025c`. Trường `provenance.code_commit` trong result giữ `9f5601d97ddf2940b32ba5e17de8d21b7c7fe1c0`, là commit preflight. Launcher so khớp 27 core-file SHA trước khi kế thừa provenance; audit độc lập xác nhận 27 SHA này giống nhau. Frozen execution commit và preflight provenance commit là hai lớp bằng chứng khác nhau.

**Không sửa bất kỳ published result nào.** Commit thực tế của Part2 sẽ được lấy từ source-manifest frozen của chính Part2 và ghi riêng trong interim bổ sung/final. Hiện Part2 chưa được launch, nên chưa có execution commit để xác nhận; không mặc định nó là 7dd9da0.

## 3. Re-score DCLM P300 parent: tiêu chí và nhánh xử lý mới

Bước này chưa chạy. Thực hiện sau P1T 3.2B kết thúc và native handoff được xác minh; không giành GPU hay ngắt run hiện tại.

- Dùng candidate frozen downstream code, cùng DCLM P300 parent identity/weights, cùng Stage1 canonical prepared requests (78,259 requests), 9 tasks, batch 2,FP32 params/BF16 AMP và cùng replay protocol.
- **PASS khi accuracy của từng task giống hệt Stage1**, không chỉ trung bình 9 task. So sánh đúng metric đã báo ở Stage1 (`acc_norm,none` hoặc `acc,none` theo bảng dưới); cũng báo các accuracy field còn lại nếu có.
- Báo **largest absolute per-item log-likelihood difference**: `max_i abs(LL_new[i] − LL_Stage1[i])`, trên toàn bộ requests đã căn chỉnh cùng task/doc/choice; kèm request/task tại cực đại. Với multiple-choice, đối chiếu từng option likelihood; không chỉ option thắng. Report chênh lệch signed tại cực đại nếu hữu ích.
- Không tự thêm điều kiện “LL phải giống bitwise” vào PASS. LL max là diagnostic phải báo; tiêu chí PASS là task accuracies giống hệt theo yêu cầu.
- Archive candidate commit, model/request/score/replay SHA, bảng task accuracy cũ/mới/difference và LL max. Hiện **LL max chưa có giá trị** vì chưa re-score.

### Nếu FAIL

**Giữ FWE downstream, vẫn tiếp tục Part2 heals.** Gửi `REPORT_interim.md` và decision interim với kết quả training đã xác minh, ghi downstream held/pending cùng bảng sai khác; không tạo downstream scores/CI giả. Sau native handoff, chạy Part2 heals trong budget đã duyệt. Nhánh này thay thế yêu cầu cũ “phải match mới tiếp tục cả Part2”. Việc thiếu downstream không được làm dừng các heals đã cho phép.

Launcher hiện có admission bắt buộc Part1 downstream bốn model đã hoàn tất. Nếu regression FAIL, phải áp dụng nhánh admission riêng có explicit failed-regression receipt + interim đã gửi + native terminal/burn locks, theo chỉ thị mới này; không giả mạo complete receipt hoặc tắt kiểm tra ownership/budget. Adaptation này chưa được GPU-deploy và không thay frozen training đang chạy.

### Accuracy tham chiếu của DCLM P300 parent (Stage1)

| Task | Metric | Accuracy tham chiếu |
|---|---|---:|
| hellaswag | `acc_norm,none` | 0.6505676160127465 |
| arc_easy | `acc_norm,none` | 0.6700336700336700 |
| arc_challenge | `acc_norm,none` | 0.3711604095563140 |
| piqa | `acc_norm,none` | 0.7568008705114254 |
| winogrande | `acc,none` | 0.6203630623520127 |
| boolq | `acc,none` | 0.6235474006116208 |
| openbookqa | `acc_norm,none` | 0.3920000000000000 |
| sciq | `acc,none` | 0.9080000000000000 |
| lambada_openai | `acc,none` | 0.6547642150203765 |

Nguồn: [Stage1 downstream results](results/a2_stage1_20261029/downstream_final/supervised/run/downstream/results.json). Prepared SHA `85cf7f2119e35e81a79aea48f58080bdc9ba5371b27accbda0eac52aa6770bbf`; original P300 scores SHA `502c080eebc32429a5539691dc9b100a39d77b3a3591a57bd95ae2c014d2db99`; replay SHA `aa01aac9553898138d07a4678d30e1d4c565c8a1b0f940403ab1a740b603dda6`.

[Evidence và hashes của audit](results/a2_stage1b_fwe_20261030/followup_audit/evidence.json). Không có GPU experiment mới được thực hiện để viết tài liệu này.

## Regression result — verified 30 October 13:33 SGT, retrieved 14:00 SGT

PASS: all nine primary task accuracies exactly match Stage 1. The maximum absolute aligned per-request log-likelihood difference is **0.0 nats** across all **78,259 requests**, with zero greedy mismatches. The complete score file is byte-identical (SHA-256 `502c080eebc32429a5539691dc9b100a39d77b3a3591a57bd95ae2c014d2db99`). Candidate execution source: `8b78d9e1051407a45be6676820bc57ac87a872e4`; original downstream source: `13686edd9cb8d7914b0f2c44c3c362c435a808f6`. Exact original model identity, prepared requests, batch 2 and FP32 parameters with BF16 AMP were retained.

| Task | Primary metric | Stage 1 | Re-score | Difference |
|---|---|---:|---:|---:|
| hellaswag | acc_norm,none | 0.650567616012746 | 0.650567616012746 | 0 |
| arc_easy | acc_norm,none | 0.67003367003367 | 0.67003367003367 | 0 |
| arc_challenge | acc_norm,none | 0.371160409556314 | 0.371160409556314 | 0 |
| piqa | acc_norm,none | 0.756800870511425 | 0.756800870511425 | 0 |
| winogrande | acc,none | 0.620363062352013 | 0.620363062352013 | 0 |
| boolq | acc,none | 0.623547400611621 | 0.623547400611621 | 0 |
| openbookqa | acc_norm,none | 0.392 | 0.392 | 0 |
| sciq | acc,none | 0.908 | 0.908 | 0 |
| lambada_openai | acc,none | 0.654764215020377 | 0.654764215020377 | 0 |

The local verifier independently recomputed the comparison from both full score arrays and the original Stage 1 summary. Native job and supervisor exited successfully; guard released; eight burn workers 722513–722520 verified with cycles 410→420 and collective payload 455.24→466.35 GiB. Raw receipt SHA-256: `98a9ebb4f9c41f481f5c4b09e9f7aa311a065a49702feed8465900757a27adb0`. [Archived evidence](results/a2_stage1b_fwe_20261030/regression_final/verification.json). FWE downstream is admitted with the same candidate evaluation/model file hashes; its results remain pending.
