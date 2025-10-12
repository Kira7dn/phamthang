# Frame Detection - Changelog

## Đơn giản hóa logic xử lý khung nhôm

### Thay đổi chính

File `utils/frame_detection.py` đã được viết lại hoàn toàn với logic đơn giản và rõ ràng hơn.

### Cấu trúc mới

#### 1. **Preprocessing** (`preprocess_image`)
- Chuyển ảnh sang grayscale
- Áp dụng OTSU threshold để tạo ảnh nhị phân

#### 2. **Loại bỏ đường kích thước** (`remove_dimension_lines`)
- Sử dụng morphology operations để phát hiện đường thẳng ngang và dọc
- Tạo mask của các đường kích thước (dimension lines)
- Loại bỏ các đường này khỏi ảnh gốc
- Làm sạch nhiễu bằng morphology close

#### 3. **Phát hiện khung bằng Contour** (`detect_frames_by_contours`)
- Tìm contours trong ảnh đã làm sạch
- Lọc theo:
  - Diện tích tối thiểu (min_area_ratio = 1% của ảnh)
  - Tỷ lệ khung hình (aspect ratio < 10)
  - Độ lấp đầy (fill_ratio > 0.5)
- Trả về danh sách các khung với thông tin: x, y, w, h, area, aspect, fill_ratio

#### 4. **Phát hiện cạnh bằng Hough Lines** (`detect_frames_by_hough`)
- Dùng Canny để phát hiện cạnh
- Áp dụng HoughLinesP để tìm đường thẳng
- Phân loại đường ngang và đường dọc dựa vào góc
- Trả về tuple của (horizontal_lines, vertical_lines) và line_image

#### 5. **Trích xuất kích thước** (`extract_frame_dimensions`)
- Kết hợp thông tin từ contours và Hough lines
- Tìm các đường thẳng nằm trong vùng khung (với margin)
- Đếm số đường ngang và dọc cho mỗi khung
- Thêm thông tin h_lines_count và v_lines_count vào mỗi frame

#### 6. **Pipeline chính** (`detect_frames_pipeline`)
Quy trình 6 bước:
1. Preprocessing → Binary image
2. Remove dimension lines → Cleaned image
3. Detect frames by contours → Frame list
4. Detect edges by Hough → Lines
5. Extract dimensions → Enhanced frames
6. Draw results → Annotated image

### So sánh với version cũ

#### Đã xóa:
- ❌ Các hàm phức tạp: `_filter_and_dedup_rects`, `dedup_rect_contours`, `remove_non_rect`
- ❌ Logic merge rectangles phức tạp với nhiều tham số
- ❌ Hàm `filter_contour` không được định nghĩa
- ❌ Hàm `detect_by_hough` không được định nghĩa
- ❌ Import các thư viện không dùng: `math`, `shutil`, `itertools.combinations`

#### Đã thêm:
- ✅ Logic rõ ràng, dễ theo dõi từng bước
- ✅ Docstrings cho tất cả các hàm
- ✅ Print thông tin chi tiết trong quá trình xử lý
- ✅ Kết hợp contour + Hough lines để trích xuất chính xác hơn
- ✅ Vẽ label với kích thước trên mỗi khung phát hiện được

### Cách sử dụng

```python
from pathlib import Path
import cv2
from utils.frame_detection import detect_frames_pipeline

# Load image
image = cv2.imread("path/to/image.png")
output_dir = Path("outputs/frame_detection")

# Run detection
annotated, frames = detect_frames_pipeline(image, output_path=output_dir)

# Access results
for frame in frames:
    print(f"Frame: {frame['w']}x{frame['h']}px")
    print(f"  Area: {frame['area']}")
    print(f"  Lines: {frame['h_lines_count']}H + {frame['v_lines_count']}V")
```

### Kết quả

Pipeline tạo ra các file trung gian để debug:
- `01_binary.png` - Ảnh nhị phân sau OTSU
- `02_cleaned.png` - Ảnh sau khi loại bỏ dimension lines
- `03_hough_lines.png` - Các đường thẳng phát hiện bởi Hough
- `10_dimension_lines_mask.png` - Mask của dimension lines
- `20_edges.png` - Cạnh phát hiện bởi Canny
- `99_final_result.png` - Kết quả cuối cùng với annotations

### Lưu ý

File cũ đã được backup tại: `utils/frame_detection_old.py`
