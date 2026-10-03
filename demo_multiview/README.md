# MVDet multi-camera demo

Demo độc lập dựa trên bố cục workspace của `D:\multicam-gt`. Giao diện chỉ đọc ảnh và annotation; occupancy map được tính bằng **inference thật từ checkpoint** đang chọn. Không cần chạy Django hoặc PostgreSQL.

## Chạy

Trên máy hiện tại, chạy:

```powershell
.\demo_multiview\run_demo.ps1
```

Mở <http://127.0.0.1:8765>. Launcher dùng dataset Wildtrack ở `D:\MVDet\downloads\wildtrack_smoke_repo\Wildtrack_dataset_full\Wildtrack_dataset` nếu đường dẫn `F:\thesis_dataset\Wildtrack_dataset` của `multicam-gt` không có. Checkpoint mặc định được đọc trực tiếp từ `C:\Users\ADMIN\Documents\mvdet_yolo26x_results.zip`, không giải nén hoặc sao chép weight vào repo.

Có thể chỉ định đường dẫn khác:

```powershell
.\demo_multiview\run_demo.ps1 `
  -WildtrackPath "F:\thesis_dataset\Wildtrack_dataset" `
  -MultiviewXPath "F:\thesis_dataset\MultiviewX_dataset" `
  -GaussianArchive "C:\Users\ADMIN\Documents\mvdet_yolo26x_results.zip"
```

Inference checkpoint hiện có cần Python với `torch`, `torchvision`, `kornia`, `numpy`, `opencv-python`, `Pillow` và GPU CUDA. Mã mô hình được nạp từ `D:\MVDet\MVDetBRL` đúng kiến trúc ghi trong manifest ZIP. Lần đầu chọn mô hình và frame, server cần thời gian nạp checkpoint, dựng chỉ mục POM và chạy GPU. Sau đó kết quả từng frame được cache trong tiến trình.

## Cách dùng

1. Chọn dataset, phương pháp và checkpoint trong thanh trên cùng.
2. Nhấn **Chạy inference**. Chuyển frame sẽ chạy checkpoint trên frame mới. Map hiển thị heatmap và các node sau NMS.
3. Nhấn node để xem bbox tham khảo và ID ở từng camera. Chọn **Annotation gốc** trong ô Phương pháp để đối chiếu với `personID` thật.

Checkpoint Gaussian hiện có chỉ hỗ trợ Wildtrack. Khi có dataset và checkpoint MultiviewX, thêm cấu hình model theo mẫu dưới đây.

## Thêm phương pháp hoặc weight

Tạo JSON registry, ví dụ `models.local.json`:

```json
[
  {
    "id": "gaussian-v2",
    "method": "Gaussian v2",
    "name": "epoch-20.pth",
    "dataset": "wildtrack",
    "backend": "mvdet",
    "arch": "resnet18",
    "threshold": 0.4,
    "weights": "D:\\models\\epoch-20.pth"
  }
]
```

Chạy `run_demo.ps1 -ModelsConfig "D:\...\models.local.json"`. Mỗi entry sẽ xuất hiện trong ô Phương pháp và Checkpoint. Với checkpoint đóng trong ZIP, thêm `"zip_member": "path/inside.zip/weight.pth"` và đặt `weights` là đường dẫn ZIP. Backend `mvdet` dùng cùng kiến trúc `PerspTransDetector`; phương pháp có kiến trúc khác cần adapter mới trong `inference.py` và đăng ký trong `PREDICTOR_BACKENDS` của `server.py`.

## Nguồn và giới hạn của bbox/ID

- **Model:** vị trí node và heatmap từ checkpoint. Bbox là phép chiếu từ `rectangles.pom` tại cùng `positionID`, không phải bbox 2D do checkpoint sinh ra. `D001`... chỉ là ID của node trong frame hiện tại, không phải ID tracking.
- **Annotation gốc:** node từ `positionID`, ID từ `personID`, bbox từ `views` của `annotations_positions/<frame>.json`.
- Giao diện chỉ chọn các frame có JSON và ảnh tại tất cả camera. Checkpoint trong ZIP được huấn luyện bằng BRL với mục tiêu Gaussian từ YOLO pseudo labels; YOLO weight không cần nạp lại khi chạy inference MVDet.

## Kiểm tra

```powershell
python -m unittest discover -s demo_multiview -p "test_*.py"
node --check demo_multiview\app.js
```

Đã đối chiếu inference trực tiếp frame 1800 với `gaussian_log/test.txt` trong ZIP: cùng 18 node, tọa độ đầu `(308, 660)`.
