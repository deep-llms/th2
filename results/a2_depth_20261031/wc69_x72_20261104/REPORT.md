# A2 Depth — X72 trên wc69

Spec SHA-256: `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`.

X72 hoàn tất 48.828 updates / 12,8B tokens, seed=0, data_seed=0; 8 GPU, LR 0,001, warmup 977, linear decay. Worker kết thúc thành công 04/11/2026 07:09:52 SGT. Kết quả được xác minh từ receipt 07:38 SGT và tải về máy trong lượt này.

| Kết quả trên final set | NLL | PPL |
|---|---:|---:|
| Parent X72 | 2.908575296108 | 18.33066419 |
| X72 S1 sau 12,8B | 2.817733402530 | 16.73886737 |

Δ_post = -0.090841893578 nats/token. Final set gồm 2.461 sequences, 10.077.795 target tokens. Đây là kết quả NLL/PPL; downstream chưa chạy.

Đã tải 23 file (3,768,152 bytes), gồm result/recipe, 3 mảng final-set parent/pre/post, 8 bộ checks, DT6, note/report và log. Đã xác minh SHA của từng file; replay độc lập 164 checks PASS: tính lại mean từ toàn bộ per-sequence arrays, đủ coverage, toàn bộ 48.828 LR updates và seed. Roundtrip difference = 0; không có flags của run X72. Checkpoint cuối 48.828 đã được kiểm tra marker và đủ 20 member sizes ở node; weights chưa tải qua kênh kết quả này.

Bản lưu đầy đủ: `artifacts/a2_depth_wc69_x72_20261104.tar.gz` (SHA-256 `22aa1275f772c239d37436d7a1de4596c5cf2fa751416ed8d34f00f916fc5247`). File manifest và independent_replay.json nằm cùng thư mục báo cáo. Scalar cũ P300/P1T không trộn vào kết quả rerun này.

P300 đang chạy; P1T còn chờ. Chưa đủ dữ liệu để kết luận E/A_init của bộ rerun hoặc paired downstream CI. Còn P300/P1T, downstream, replay tổng, báo cáo cuối và native handoff. Giữ các flags mất weights node cũ, resume mới được kiểm tra trên CPU, Breadth-before-Depth và X72-first theo yêu cầu user, cùng toàn bộ lỗi lịch sử.
