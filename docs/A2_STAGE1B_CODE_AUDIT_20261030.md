# Stage1b code/provenance audit trước interim

Đối chiếu ngày 30/10/2026. P1T 3.2B tiếp tục chạy; audit này chỉ dùng Git và receipt đã tải về, không chạy GPU.

## 1. Training: 0c1c148 → 7dd9da0

Hai manifest frozen có **103 → 124 file: 99 file giữ nguyên, 4 sửa, 21 thêm; không xóa**. Đây là tập file snapshot chạy trên node, không phải tổng số file trong Git. Toàn bộ diff Git gồm **162 file**, bao gồm docs, kết quả, test và locks; danh sách từng file nằm trong phụ lục bên dưới và [audit.json](results/a2_stage1b_fwe_20261030/code_audit/audit.json).

`efficient_llm/a2/{model,train,measure,data,protocol,check}.py`, toàn bộ `efficient_llm/a2_stage1/*.py`, `packed_training.py`, `packed_evaluation.py` và Accelerate config không đổi byte trong hai snapshot. Forward, removal 16–21, KL, AdamW, lịch LR, precision và primitive scoring vẫn là implementation cũ. Tuy nhiên có adapter đi qua đường chọn model/data/evaluation và sửa admission data; không coi toàn bộ thay đổi là “chỉ orchestration”.

### Từng file trong frozen delta

A = thêm; M = sửa. “Có” nghĩa file nằm trên đường cấu hình/thực thi số; mô tả phân biệt đổi công thức với đổi dispatch/admission.

| Loại | File | Chạm đường numerical? | Thay đổi |
|---|---|---|---|
| A | `docs/A2_stage1b_fwe_spec.md` | Không thực thi | Pre-registration FWE v1.1, thứ tự Part1/Part2 và giới hạn ngân sách. |
| A | `efficient_llm/a2_stage1b/__init__.py` | Không | Khởi tạo package. |
| A | `efficient_llm/a2_stage1b/model_stage.py` | Có: chọn model/data, gọi training và evaluation | Adapter từ Stage1: chọn FWE parents/teacher, held-out và stream; kiểm tra spec. Forward, prune(range(16,22)), KD/KL, optimizer/schedule, AMP và score gọi implementation cũ, không đổi công thức. Nhánh T17 giữ nem/CulturaX gốc. |
| A | `efficient_llm/a2_stage1b/protocol.py` | Có: cấu hình run/LR/budget | Kế thừa Stage1 Run, GRID, grid_choice, decision; đổi run ID/dataset và thứ tự hai phase. Không đổi hàm lịch LR. |
| M | `eval/a2_checkpoint.py` | Có: đường load/identity của evaluation | Dispatch schema2 sang bộ xác minh checkpoint Stage1; không sửa forward/scoring. |
| M | `eval/a2_run.py` | Có: orchestration evaluation/scoring | Tách _run_screen dùng validator/gate_names/finalize; giữ scorer, requests, precision và replay. Thay đổi này đã có trong downstream Stage1 13686ed. |
| A | `eval/a2_stage1_checkpoint.py` | Có: chọn/load checkpoint, xác minh layer removal | Ràng buộc native export với receipt; kiểm tra 18 layer/kept_layers; thêm dataset FWE. Không thay đổi phép prune hay forward. |
| A | `eval/a2_stage1_plan.py` | Có: chọn model evaluation | Chọn sáu identity Stage1 DCLM, kiểm tra terminal và SHA; không tính likelihood. |
| A | `eval/a2_stage1_run.py` | Có: orchestration evaluation | Sáu model, hai GPU gate, shared scorer và paired statistics; đã dùng trong Stage1 downstream. |
| A | `eval/a2_stage1_stats.py` | Có: thống kê evaluation | Paired bootstrap 10k seed0, căn chỉnh replay; mở rộng model/contrast tham số cho 4+2. Đây là mã số thống kê, không phải training. |
| A | `eval/a2_stage1b_plan.py` | Có: chọn model evaluation | Terminal/identity locks FWE theo phase, 4 model Part1 và 2 Part2. |
| A | `eval/a2_stage1b_run.py` | Có: orchestration evaluation/scoring | Adapter 4+2, batch2, 9 task, hai gate mỗi phase, gọi shared _run_screen. |
| A | `eval/a2_stage1b_stats.py` | Có: thống kê evaluation | Đặt model/contrast cho Part1/Part2/final; gọi bootstrap dùng chung. |
| A | `resources/a2_stage1b_fwe_release_20261030.json` | Không thực thi; khóa dữ liệu | Receipt release/revision và SHA của 60 source shard FWE. |
| M | `scripts/a2_stage1_data.py` | Có: đường data packing/order | Tham số hóa corpus; thêm dùng receipt full-hash trước đó để tránh rehash nguồn; thêm khóa replication spec. Thuật toán shuffle seed2030/copy token/packing giữ nguyên. |
| A | `scripts/a2_stage1b.py` | Có: chọn run/data, LR và tổng hợp evaluation | Coordinator FWE; grid/edge/commonLR, Part1/Part2, validates full schedule, computes E/A. Dùng grid_choice/validate_fit/report_runs cũ; không sửa train loop. |
| A | `scripts/a2_stage1b_budget.py` | Không | Ledger/admission 20h report,24h hard, deadline/cleanup. |
| A | `scripts/a2_stage1b_data.py` | Có: chọn dữ liệu/packing | Admission FWE đã pin, kiểm tra release/spec, gọi prepare dùng chung với corpus=fineweb_edu. |
| A | `scripts/a2_stage1b_preflight.py` | Không tính forward/loss; tổng hợp kết quả số | Gộp T1/T3/T7/T8/T17/T18, eligibility và budget; không tự chấm model. |
| M | `scripts/launch_a2_downstream.py` | Không | freeze_source nhận extra_files; thay đổi staging/manifest, không đổi numerical code. |
| A | `scripts/launch_a2_stage1_downstream.py` | Có: cấu hình evaluation, không tính score | Launcher Stage1: freeze/identity/budget/native supervisor, fixed six-model plan. |
| A | `scripts/launch_a2_stage1b_data.py` | Có: cấu hình data job, không tự packing | CPU shuffle launcher, frozen source và paths; gọi preparation đã mô tả. |
| A | `scripts/launch_a2_stage1b_downstream.py` | Có: cấu hình evaluation/precision env | Launcher 4+2; identity locks, batch2, Accelerate/env, native handoff. Không sửa scoring kernel. |
| A | `scripts/launch_a2_stage1b_heals.py` | Có: cấu hình training/data/precision env | Chọn phase, kiểm tra 27 core SHA bằng preflight, giữ provenance kiểm thử, Accelerate8/BF16, frozen launch; không đổi optimizer hay forward. |
| A | `scripts/launch_a2_stage1b_preflight.py` | Có: cấu hình model/data/tests/precision env | Admission FWE/parent/T3; tạo inputs, 27 core SHA, jobs T17 và các gate; giữ native supervisor. |

### Cùng snapshot cho các run?

**Có:** 8 grid runs, P300 3.2B và P1T 3.2B đang chạy đều thuộc cùng Part1 root/source snapshot `7dd9da088ba5af581c295775819319ce54b2025c`, native supervisor715452. Chín result đã có và process chain của P1T được kiểm tra tại 11:01 SGT; P1T chưa có result cuối.

**Phân biệt hai trường:** `source-manifest.code_commit` là `7dd9da0`; `result.provenance.code_commit` giữ **`9f5601d97ddf2940b32ba5e17de8d21b7c7fe1c0`**, commit preflight. Launcher đối chiếu SHA cả 27 file core, specs/data/assets/runtime/microbatch/chunk trước khi kế thừa provenance. Audit đã kiểm tra lại 27 SHA bằng manifest Part1 và Git7dd9da0, cùng provenance của cả 9 result đã công bố. Không được đọc trường provenance đó thành toàn bộ source Part1 chạy tại9f5601d.

### T17 đã chạy lại trước FWE

T17 chạy bằng adapter Stage1b tại preflight9f5601d, kết thúc **04:04:53 SGT**, trước scientific training bắt đầu **04:19:10 SGT**. 27 core files của preflight và Part1 giống byte.

- Run gốc/rerun: `P0-nem-P300-S1-0.05B-lr3e-4`, **CulturaX stream gốc**, data SHA khớp Phase0A; không dùng FWE stream cho T17.
- 190 update, 64×4096 token/update (49,807,360 token), 8 rank đúng thứ tự, full LR schedule, cùng precision procedure và parent/teacher/layer set. Roundtrip difference0.
- Original final NLL **3.061875752041902**; rerun **3.0617311959690636**.
- Difference **−0.0001445560728385864 nats**; **PASS**, absolute difference <0.002.
- Original T17 receipt SHA `9056d874791badc922103a3589a9b49ced3554d4a54f20403cd02e1ae6a73188`; tests receipt SHA `0ddedf66933134338cc87f872b28e5df9d8a67ba0cf6460d482c90ebe0321b8b`.

[Compact T17 receipt](results/a2_stage1b_fwe_20261030/preflight_final/supervised/run/t17/T17.json). Compact archive có thể khác hash bản gốc; audit đối chiếu bản gốc trong receipt tải về. Điều kiện “chưa chạy T17 trước FWE” không xảy ra, nên không cần chạy lại T17 sau P1T chỉ vì câu hỏi này.

## 2. Downstream

**Không dùng nguyên commit13686ed.** Adapter FWE4+2 được thêm trong7dd9da0; chưa GPU-deploy downstream FWE và chưa chốt commit launch cuối. Không khẳng định equivalence chỉ từ CPU tests.

So với13686ed, trong `eval/` chỉ có hai file sửa (`a2_stage1_checkpoint.py`: FWE checkpoint identity; `a2_stage1_stats.py`: model/contrast tham số) và ba file thêm (`a2_stage1b_plan.py`, `a2_stage1b_run.py`, `a2_stage1b_stats.py`). Chín file scoring/request/replay/gate liệt kê trong audit.json giống byte. Điều này hỗ trợ static review nhưng chưa thay thế regression trên GPU.

> **Cập nhật theo follow-up mới:** Chính sách gate trong đoạn dưới là lịch sử và đã được thay thế bởi [câu trả lời follow-up, mục 3](A2_STAGE1B_FOLLOWUP_REPLY_20261030.md). PASS yêu cầu accuracy từng task giống hệt; LL max là diagnostic. Nếu FAIL: giữ FWE downstream, gửi interim không có downstream và tiếp tục Part2 heals sau native handoff trong ngân sách. Không dừng P1T.

**Gate mới theo yêu cầu user:** ngay sau P1T3.2B terminal + independent native burn handoff, trước FWE downstream và Part2, re-score **DCLM P300 parent** với candidate frozen downstream code, exact Stage1 identity/weights, canonical prepared requests và protocol (78,259 requests;9 tasks;batch2;FP32 parameters/BF16 AMP). CPU replay và đối chiếu per-request outputs cùng per-task metrics với Stage1. Archive đầy đủ identity/source/request/result SHA và sai khác; chỉ tiếp tục khi đã xác minh match. Nếu mismatch, báo rõ và xử lý trước khi dùng downstream FWE. Bước này **chưa chạy**, được đưa vào ledger24h/report20h. Job P1T hiện tại không bị dừng.

## 3. Spec hashes thực tế

Đã kiểm tra trong original preflight/tests, inputs Part1 và cả9 published result receipts:

| Trường provenance | SHA-256 |
|---|---|
| `stage1b_spec_sha256` (v1.1) | `b2121a4f2859f99db421273082c6be035dc5e833af6d1d071f5ed4cd381ed483` |
| `spec_sha256` (Stage1 v3.3) | `9ebe0d0a377f0107f7058bf5e4899b5380cbe5f4f518b09091ac533d2c9f254e` |

Cả hai đúng theo yêu cầu. P1T đang chạy dùng cùng inputs/spec locks; kiểm tra lại result cuối khi xuất hiện. `base_spec_sha256` là spec Phase0A khác, không thay thế `spec_sha256` v3.3.

## Nội dung phải đưa vào REPORT_interim.md

- Heal stream: `HuggingFaceFW/fineweb-edu`, prepared revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Release gốc `v1.0.0`, revision `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`; cả60 selected shard có cùng byte/LFS SHA ở hai revision theo [release receipt](../resources/a2_stage1b_fwe_release_20261030.json). Đây là release parents dùng; “1.4T” là quy mô corpus, không phải version1.4.0.
- T1: cảP300/P1T/S72 PASS; FP32–AMP gaps lần lượt0.000091346475/0.000156757084/0.000077693377.
- T3 PASS, không có reason lỗi; S72 đủ điều kiện.
- T7 PASS, document-id overlap0 và content-hash overlap0.
- T18:7,324,217 sequences;29,155,864 EOS;EOS share0.000971862366; parent FWE held-out loss thấp hơn CulturaX ở cả3parent, không flag. Kèm first/last decode trong data verification.

## Phụ lục: toàn bộ Git diff 0c1c148 → 7dd9da0

Mỗi file dưới đây có trạng thái và phân loại. Các file không thuộc frozen delta không được dùng để suy ra “103→124”. Tests/docs/receipt chỉ là evidence hoặc kiểm thử; resources khóa identity được ghi rõ.

| Loại | File | Chạm numerical path / vai trò |
|---|---|---|
| M | `docs/A2_CORPORA_30B_20261029.md` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| M | `docs/A2_STAGE1_EXECUTION_20261029.md` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/A2_STAGE1_PREFLIGHT_20261029.md` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/A2_STAGE1_RESULTS_20261030.md` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/A2_stage1b_fwe_spec.md` | Không thực thi; Pre-registration FWE v1.1, thứ tự Part1/Part2 và giới hạn ngân sách. |
| M | `docs/CURRENT_TASK.md` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| M | `docs/PROJECT_NOTES.md` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_corpora_20261029/fwe_final/corpora_complete.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_corpora_20261029/fwe_final/fwe_manifest.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_corpora_20261029/fwe_final/verification.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/data_verification/complete.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/data_verification/manifest.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/data_verification/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0003/gpus-free-before-training.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0003/identities.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0003/plan.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0003/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0003/reclaim.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0003/stage-budget.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0017/gpus-free-before-training.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0017/identities.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0017/plan.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0017/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0017/reclaim.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_0017/stage-budget.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/SHA256SUMS.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/identities/P1T-parent-identity.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/identities/P1T-student-identity.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/identities/P300-parent-identity.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/identities/P300-student-identity.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/identities/S72-parent-identity.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/identities/X72-student-identity.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/identities/identities.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/independent_replay_verification.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/jobs.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/plan.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/raw_receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/source-manifest.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/stage-budget.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/supervised/burn-verified.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/supervised/gpus-free-before-training.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/supervised/reclaim.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/supervised/run/complete.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/supervised/run/downstream/P300-parent-check.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/supervised/run/downstream/P300-student-check.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/supervised/run/downstream/complete.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/supervised/run/downstream/paired_accuracy.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/supervised/run/downstream/results.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/supervised/run/run.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/supervised/supervisor.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/downstream_final/verification.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/final/REPORT.md` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/final/SHA256SUMS.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/final/compute.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/final/decision_stage1.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/final/parents_summary.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/final/runs_summary.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/final/secondary_diagnostics.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/historical_comparison.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/preflight_0859/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/preflight_final/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_0947/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1017/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1047/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1117/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1147/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1217/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1247/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1318/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1349/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1418/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1448/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1523/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1617/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1647/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1717/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1747/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1817/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1847/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1917/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_1947/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_2017/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_2047/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_2117/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_2148/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_2217/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_2247/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_2317/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/status_2348/receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/training_final/burn.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/training_final/choice.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/training_final/complete.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/training_final/decision.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/training_final/run_complete.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/training_final/supervisor.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/training_final/tests_receipt.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1_20261029/training_final/verification.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/data_0339/verification.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/data_final/complete.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/data_final/gpu-inspection.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/data_final/manifest.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/data_final/source-manifest.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/data_final/verification.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/inventory/verification.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_0401/verification.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/SHA256SUMS` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/inputs.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/jobs.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/source-manifest.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/stage-budget.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/supervised/burn-verified.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/supervised/run/P1T.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/supervised/run/P300.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/supervised/run/S72.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/supervised/run/complete.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/supervised/run/numerical.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/supervised/run/t17/T17.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/supervised/run/tests.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/supervised/run/throughput.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/supervised/supervisor.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `docs/results/a2_stage1b_fwe_20261030/preflight_final/verification.json` | Không thực thi; Tài liệu hoặc receipt kết quả; không thay đổi numerical path. |
| A | `efficient_llm/a2_stage1b/__init__.py` | Không; Khởi tạo package. |
| A | `efficient_llm/a2_stage1b/model_stage.py` | Có: chọn model/data, gọi training và evaluation; Adapter từ Stage1: chọn FWE parents/teacher, held-out và stream; kiểm tra spec. Forward, prune(range(16,22)), KD/KL, optimizer/schedule, AMP và score gọi implementation cũ, không đổi công thức. Nhánh T17 giữ nem/CulturaX gốc. |
| A | `efficient_llm/a2_stage1b/protocol.py` | Có: cấu hình run/LR/budget; Kế thừa Stage1 Run, GRID, grid_choice, decision; đổi run ID/dataset và thứ tự hai phase. Không đổi hàm lịch LR. |
| M | `eval/a2_checkpoint.py` | Có: đường load/identity của evaluation; Dispatch schema2 sang bộ xác minh checkpoint Stage1; không sửa forward/scoring. |
| M | `eval/a2_run.py` | Có: orchestration evaluation/scoring; Tách _run_screen dùng validator/gate_names/finalize; giữ scorer, requests, precision và replay. Thay đổi này đã có trong downstream Stage1 13686ed. |
| A | `eval/a2_stage1_checkpoint.py` | Có: chọn/load checkpoint, xác minh layer removal; Ràng buộc native export với receipt; kiểm tra 18 layer/kept_layers; thêm dataset FWE. Không thay đổi phép prune hay forward. |
| A | `eval/a2_stage1_plan.py` | Có: chọn model evaluation; Chọn sáu identity Stage1 DCLM, kiểm tra terminal và SHA; không tính likelihood. |
| A | `eval/a2_stage1_run.py` | Có: orchestration evaluation; Sáu model, hai GPU gate, shared scorer và paired statistics; đã dùng trong Stage1 downstream. |
| A | `eval/a2_stage1_stats.py` | Có: thống kê evaluation; Paired bootstrap 10k seed0, căn chỉnh replay; mở rộng model/contrast tham số cho 4+2. Đây là mã số thống kê, không phải training. |
| A | `eval/a2_stage1b_plan.py` | Có: chọn model evaluation; Terminal/identity locks FWE theo phase, 4 model Part1 và 2 Part2. |
| A | `eval/a2_stage1b_run.py` | Có: orchestration evaluation/scoring; Adapter 4+2, batch2, 9 task, hai gate mỗi phase, gọi shared _run_screen. |
| A | `eval/a2_stage1b_stats.py` | Có: thống kê evaluation; Đặt model/contrast cho Part1/Part2/final; gọi bootstrap dùng chung. |
| A | `resources/a2_stage1_reference_20261029.json` | Không thực thi; khóa identity; Terminal/reference lock cho downstream; không sửa công thức số. |
| A | `resources/a2_stage1_terminal_20261029.json` | Không thực thi; khóa identity; Terminal/reference lock cho downstream; không sửa công thức số. |
| A | `resources/a2_stage1b_fwe_release_20261030.json` | Không thực thi; khóa dữ liệu; Receipt release/revision và SHA của 60 source shard FWE. |
| M | `scripts/a2_stage1_data.py` | Có: đường data packing/order; Tham số hóa corpus; thêm dùng receipt full-hash trước đó để tránh rehash nguồn; thêm khóa replication spec. Thuật toán shuffle seed2030/copy token/packing giữ nguyên. |
| A | `scripts/a2_stage1b.py` | Có: chọn run/data, LR và tổng hợp evaluation; Coordinator FWE; grid/edge/commonLR, Part1/Part2, validates full schedule, computes E/A. Dùng grid_choice/validate_fit/report_runs cũ; không sửa train loop. |
| A | `scripts/a2_stage1b_budget.py` | Không; Ledger/admission 20h report,24h hard, deadline/cleanup. |
| A | `scripts/a2_stage1b_data.py` | Có: chọn dữ liệu/packing; Admission FWE đã pin, kiểm tra release/spec, gọi prepare dùng chung với corpus=fineweb_edu. |
| A | `scripts/a2_stage1b_preflight.py` | Không tính forward/loss; tổng hợp kết quả số; Gộp T1/T3/T7/T8/T17/T18, eligibility và budget; không tự chấm model. |
| M | `scripts/launch_a2_downstream.py` | Không; freeze_source nhận extra_files; thay đổi staging/manifest, không đổi numerical code. |
| A | `scripts/launch_a2_stage1_downstream.py` | Có: cấu hình evaluation, không tính score; Launcher Stage1: freeze/identity/budget/native supervisor, fixed six-model plan. |
| A | `scripts/launch_a2_stage1b_data.py` | Có: cấu hình data job, không tự packing; CPU shuffle launcher, frozen source và paths; gọi preparation đã mô tả. |
| A | `scripts/launch_a2_stage1b_downstream.py` | Có: cấu hình evaluation/precision env; Launcher 4+2; identity locks, batch2, Accelerate/env, native handoff. Không sửa scoring kernel. |
| A | `scripts/launch_a2_stage1b_heals.py` | Có: cấu hình training/data/precision env; Chọn phase, kiểm tra 27 core SHA bằng preflight, giữ provenance kiểm thử, Accelerate8/BF16, frozen launch; không đổi optimizer hay forward. |
| A | `scripts/launch_a2_stage1b_preflight.py` | Có: cấu hình model/data/tests/precision env; Admission FWE/parent/T3; tạo inputs, 27 core SHA, jobs T17 và các gate; giữ native supervisor. |
| A | `tests/a2/test_stage1_checkpoint.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1_eval_launcher.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1_eval_plan.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1_eval_run.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1_plan.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1_stats.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1b_checkpoint.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1b_data.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1b_eval_launcher.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1b_eval_plan.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1b_eval_run.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1b_launch.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1b_parts.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1b_preflight.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
| A | `tests/a2/test_stage1b_stats.py` | Không trong production path; CPU test; kiểm tra các adapter tương ứng, không được freeze vào 124-file snapshot. |
