# A2 Depth: tài liệu cần bổ sung để tiếp tục

Cập nhật: 01/11/2026, SGT.

**ĐÃ GIẢI QUYẾT:** user đã cung cấp [A2_depth_spec.md](A2_depth_spec.md) v1.1, SHA-256 `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`. Nội dung bên dưới là bản ghi lịch sử trước khi nhận spec; không còn yêu cầu user bổ sung tài liệu. Depth A → DT4/B → DT6/C/downstream được triển khai theo spec.

## Thiếu gì, và vì sao nhắc đến Depth?

**Thiếu bản đặc tả `docs/A2_depth_spec.md` v1.1 mà tài liệu Breadth tham chiếu.** Đây không phải phần còn thiếu của thí nghiệm Breadth.

Trong [A2_breadth_spec.md](A2_breadth_spec.md), dòng 6 yêu cầu chạy Breadth sau báo cáo Depth Parts A–B, rồi tiếp tục Depth Part C, bắt đầu bằng DT6. Mục 4, bước 5 cũng yêu cầu tiếp tục A2 D Part C. Tuy nhiên, tài liệu Breadth không định nghĩa các thí nghiệm A–B, Part C hay DT6.

Đã kiểm tra trong checkout hiện tại:

- Không có `docs/A2_depth_spec.md`.
- Không tìm thấy tài liệu có tên chứa `depth` dạng Markdown, text hoặc PDF trong các file dự án được tìm kiếm.
- Không có lịch sử của đường dẫn `docs/A2_depth_spec.md` trong các Git ref hiện có tại máy dev.

Điều này chỉ xác nhận tài liệu chưa có trong workspace/lịch sử local đã kiểm tra; không khẳng định tài liệu không tồn tại ở nơi khác.

## Cần bạn cung cấp

**Chỉ cần bổ sung bản gốc `A2_depth_spec.md` v1.1 vào `docs/`, hoặc chỉ đường dẫn/commit chứa bản đó.** Không cần viết lại các thông số nếu chúng đã nằm trong bản gốc.

Nếu Parts A–B đã chạy ở nơi khác, cần thêm đường dẫn báo cáo và kết quả của chúng để đối chiếu điều kiện trước Part C. Hiện chưa xác minh được trạng thái Parts A–B; không coi là đã hoàn thành, cũng không kết luận chúng chưa từng chạy.

Nếu không còn bản đặc tả gốc, tài liệu thay thế cần xác định tối thiểu:

| Nội dung | Thông tin cần có |
|---|---|
| Phạm vi và thứ tự | Parts A–B và Part C gồm những thí nghiệm nào; quan hệ phụ thuộc giữa chúng |
| DT6 | DT6 kiểm tra điều gì, thực hiện bằng cách nào, tiêu chí đánh giá và kết quả cần lưu |
| Model và pruning | Checkpoint/parent được chọn, dataset, các lớp hoặc block bị loại, cách đánh số lớp |
| Recipe | Có healing/distillation hay chỉ đánh giá; nếu có training thì teacher, số token/update, LR và những khác biệt so với recipe A2 đã có |
| Đánh giá | Split và metric, định nghĩa các tương phản, bootstrap/seed nếu khác quy ước A2 đã có |
| Kết luận và đầu ra | Các giả thuyết/điều kiện quyết định đã định trước, file kết quả và báo cáo cần xuất |

Đây là danh sách thông tin để khôi phục đúng kế hoạch nghiên cứu, không phải đề xuất thêm thí nghiệm. Các quy tắc đã chốt về kiểm tra scientific/time theo chế độ ghi nhận và tiếp tục, quyền sở hữu GPU, cùng worker nhắc kiểm tra vẫn được giữ nguyên.

## Những gì đã có, không cần cung cấp lại

- [Báo cáo Breadth hoàn chỉnh](../results/a2_breadth_20261031/REPORT.md): 4 lượt heal, 44 cấu hình SmolLM3, đủ 56 file NLL.
- 159 file kết quả đã tải về dev, kiểm tra checksum và replay CPU độc lập; bàn giao GPU đã được xác minh.
- Code, cấu hình, manifest và các kết quả A2 trước đó đang có trong dự án. Khi có đặc tả Depth, agent sẽ tự đối chiếu và tái sử dụng các tài nguyên phù hợp.
- Breadth không cần chạy lại vì thiếu tài liệu Depth. Sai lệch thứ tự so với tham chiếu Depth đã được ghi trong báo cáo.

## Bước tiếp theo

Khi có đặc tả, agent sẽ đọc và ghi SHA, xác định phần nào đã có kết quả, rồi triển khai các bước còn lại trong phạm vi được user ủy quyền. Mọi job dài sẽ có worker nhắc trong chính thread này, được xác minh hoạt động trước khi kết thúc lượt khởi chạy; không tạo watcher trùng.

**Hiện tại:** Breadth đã hoàn tất; chưa khởi chạy Depth C/DT6 vì chưa có định nghĩa của chúng. Worker Breadth đã dừng sau khi hoàn tất, không có job Depth đang chờ worker theo dõi.
