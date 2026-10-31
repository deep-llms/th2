# A2 Q — sửa QT3 và chấp nhận deviation QT6

User chấp nhận kết quả A2 Q ngày 31/10/2026 và yêu cầu follow-up chỉ kiểm tra trọng số đã lưu, không chạy quantization mới. Đây là amendment cho spec gốc SHA256 `a7fb04042d54f977cff4ea5c2275b6b2af3a7c00cce4b2898ffc3dc9383c44da`; file spec và mọi kết quả gốc giữ nguyên.

QT3 sửa đổi kiểm tra mọi matrix và mọi group 128 cột:

- Số giá trị phân biệt không vượt quá `2^b`.
- Dùng grid đã tạo ra trọng số để kiểm tra mã sau khi quantize lại ŵ. RTN dựng lại grid từ parent, với đúng phép tính FP32 của implementation GPU gốc.
- GPTQ dùng grid đã lưu nếu có. Lần chạy này không lưu grid riêng: dùng fallback user cho phép, kiểm tra các giá trị phân biệt nằm trên một uniform grid trong sai số làm tròn BF16. Fallback không khẳng định khôi phục được grid/codes GPTQ ban đầu.

QT3 gốc refit min/max trên ŵ vẫn lưu nguyên như lịch sử. Với asymmetric grid và zero-point làm tròn, việc refit có thể tạo grid khác; làm tròn BF16 cũng có thể thay đổi kết quả. QT4 và QT5 đã PASS và giữ nguyên. Chỉ bỏ `unverified` do QT3 cho phần audit mới đạt; không xóa các nguyên nhân độc lập khác.

QT6 vẫn ghi sự kiện một document `train25978874` nằm qua ranh giới cal0/cal1, không trùng tuning hoặc final. User chấp nhận đây là **minor deviation**, không rerun và không tiếp tục đánh dấu `unverified` chỉ vì QT6. Cờ sensitivity của contrast tính từ chênh lệch cal0–cal1 là phép đo riêng, giữ nguyên.

Audit chạy trên CPU, không forward, không gọi routine RTN/GPTQ, không sửa trọng số hoặc chạm workload GPU. Lượt audit đầu dùng phép chia CPU trực tiếp cho scale RTN; CUDA tối ưu tensor chia CPU-scalar bằng nhân nghịch đảo FP32. Lượt kiểm tra lại riêng RTN dùng đúng phép tính này. Xem [PyTorch kernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu). Cả hai lượt audit được lưu để truy vết.

Bằng chứng và trạng thái từng run được cập nhật tại [báo cáo A2 Q](../results/a2_quant_20261031/REPORT.md).
