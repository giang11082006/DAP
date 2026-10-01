# FEATURES.md — Mô tả feature của training_dataset.csv

Tài liệu mô tả các cột trong `data/features/training_dataset.csv`, được tạo bởi `src/pipeline.py`.
Bài toán: dự đoán giá cà phê Đắk Lắk của **ngày kế tiếp** (`next_day_price`) từ dữ liệu giá và thời tiết tới hết ngày hiện tại.

Người phụ trách: Member 3 (Data Processing / Feature Engineering).

## 1. Luồng xử lý

1. **Đọc dữ liệu raw** (không sửa file gốc):
   - Giá cà phê: `data/raw/coffee/coffee_daklak_3y_complete.csv`
   - Thời tiết: `data/raw/weather/weather.csv` (Open-Meteo, bỏ 2 dòng metadata đầu file)
2. **Cleaning**:
   - Chuyển `date` sang kiểu ngày, `price` và các cột thời tiết sang kiểu số (giá trị lỗi thành NaN).
   - Bỏ dòng thiếu `date` hoặc `price` (với giá cà phê).
   - Xóa trùng theo `date` (giữ dòng cuối), sắp xếp theo thời gian.
   - Đổi tên cột thời tiết cho gọn (bỏ đơn vị trong tên cột).
3. **Merge**: ghép giá cà phê và thời tiết theo `date` bằng `inner join`, chỉ giữ những ngày có đủ cả hai nguồn.
4. **Feature engineering**: tạo lag, rolling, calendar và target (mục 3).
5. **Dọn NaN**: bỏ các dòng có NaN do `shift` và `rolling` (khoảng 30 ngày đầu do cần lịch sử, và dòng cuối do chưa có target).

File sinh ra: `data/processed/coffee_clean.csv`, `weather_clean.csv`, `merged_daily.csv` và `data/features/training_dataset.csv`.

## 2. Cột gốc (sau khi merge)

| Cột | Ý nghĩa | Đơn vị |
|---|---|---|
| `date` | Ngày quan sát | yyyy-mm-dd |
| `coffee_price` | Giá cà phê của ngày hiện tại | VND/kg |
| `temperature_2m_max` | Nhiệt độ cao nhất trong ngày | °C |
| `temperature_2m_min` | Nhiệt độ thấp nhất trong ngày | °C |
| `temperature_2m_mean` | Nhiệt độ trung bình | °C |
| `precipitation_sum` | Tổng lượng giáng thủy | mm |
| `rain_sum` | Tổng lượng mưa | mm |
| `precipitation_hours` | Số giờ có mưa | giờ |
| `relative_humidity_2m_mean` | Độ ẩm tương đối trung bình | % |
| `soil_moisture_0_to_7cm` | Độ ẩm đất lớp 0–7 cm | m³/m³ |
| `soil_moisture_7_to_28cm` | Độ ẩm đất lớp 7–28 cm | m³/m³ |
| `shortwave_radiation_sum` | Tổng bức xạ sóng ngắn | MJ/m² |

## 3. Feature được tạo thêm

### 3.1 Lag giá (giá của các ngày trước)

| Cột | Cách tính |
|---|---|
| `price_lag_1`, `price_lag_2`, `price_lag_3` | Giá cách 1, 2, 3 ngày trước |
| `price_lag_7`, `price_lag_14`, `price_lag_30` | Giá cách 7, 14, 30 ngày trước |

Tính bằng `price.shift(lag)` với lag dương, nên chỉ lấy dữ liệu quá khứ.

### 3.2 Rolling giá

| Cột | Cách tính |
|---|---|
| `price_mean_7`, `price_mean_14`, `price_mean_30` | Trung bình giá của 7/14/30 ngày trước đó |
| `price_std_7`, `price_std_30` | Độ lệch chuẩn giá của 7/30 ngày trước đó (độ biến động) |

Cửa sổ được tính sau `shift(1)`, nên **không bao gồm giá của chính ngày hiện tại**.

### 3.3 Rolling thời tiết

| Cột | Cách tính |
|---|---|
| `rain_7d`, `rain_30d` | Trung bình `rain_sum` của 7/30 ngày trước đó |
| `temperature_mean_7d`, `temperature_mean_30d` | Trung bình `temperature_2m_mean` của 7/30 ngày trước đó |
| `humidity_mean_7d`, `humidity_mean_30d` | Trung bình `relative_humidity_2m_mean` của 7/30 ngày trước đó |

Cũng dùng `shift(1)` trước khi `rolling`. Thời tiết tích lũy trong vài tuần có thể ảnh hưởng đến mùa vụ và giá, nên dùng cửa sổ dài.

### 3.4 Calendar

| Cột | Cách tính |
|---|---|
| `day_of_week` | Thứ trong tuần (0 = Thứ Hai … 6 = Chủ Nhật) |
| `day_of_month` | Ngày trong tháng (1–31) |
| `month` | Tháng (1–12) |
| `quarter` | Quý (1–4) |

Lưu ý: hai cột `sin_month` và `cos_month` (mã hóa tuần hoàn của tháng) đang được **comment trong code** nên hiện không có trong dataset.

### 3.5 Target và cột kiểm tra

| Cột | Vai trò | Cách tính |
|---|---|---|
| `next_day_price` | **Target** | `coffee_price.shift(-1)`, tức giá của ngày kế tiếp |
| `date_gap_days` | Cột kiểm tra, **không dùng làm feature** | Số ngày giữa ngày hiện tại và ngày kế tiếp, để phát hiện dữ liệu bị đứt quãng |

## 4. Kiểm tra data leakage

| Nguy cơ | Cách xử lý |
|---|---|
| Lag lấy nhầm giá tương lai | Chỉ dùng `shift` với số dương (quá khứ) |
| Rolling bao gồm giá/thời tiết của ngày hiện tại hoặc tương lai | Tính trên chuỗi đã `shift(1)` |
| Target bị lẫn vào feature | Target là `shift(-1)`; không có cột feature nào dùng giá ngày kế tiếp |
| Shuffle làm lộ tương lai | Dữ liệu luôn sắp xếp theo `date`; việc chia train/val/test theo thứ tự thời gian được thực hiện ở bước modeling |
| `date_gap_days` dùng ngày kế tiếp | Chỉ để kiểm tra chất lượng dữ liệu, cần loại khỏi tập feature khi train |

Các feature dùng giá và thời tiết của chính ngày hiện tại (`coffee_price`, `temperature_2m_*`, `rain_sum`, ...) là hợp lệ, vì khi dự đoán giá ngày mai thì các giá trị này của hôm nay đã biết.

## 5. Giới hạn đã biết

- Merge dạng `inner`: những ngày có giá nhưng chưa có thời tiết (cuối khoảng dữ liệu) bị loại.
- Dataset phù hợp cho dự đoán một bước (next day). Chưa có target cho dự đoán 3 ngày hoặc 7 ngày.
- Các ngày đầu chuỗi bị loại vì chưa đủ lịch sử cho `price_lag_30` và rolling 30 ngày.
