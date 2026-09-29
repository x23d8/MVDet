# MVDet pixel-wise max, 2 GPU

Notebook này chạy ablation không cluster các điểm chân YOLO theo bán kính. Mỗi
detection `person` của YOLO26x được chiếu từ bottom-center của bbox xuống BEV;
Gaussian của các detection/camera được hợp nhất bằng pixel-wise maximum khi tạo
pseudo loss. Nhờ vậy hai người đứng gần nhau không bị gộp thành một tâm trước
khi MVDet học.

Cấu hình mặc định là `drop=60`, `lambda_pseudo=0.01`, 10 epoch, global batch 2.
`torch.nn.DataParallel` chia batch thành một frame (gồm đủ C1-C7) trên mỗi T4.
YOLO pseudo-label generation chạy trước trên GPU 0 với batch 1; giai đoạn này
không chạy đồng thời với MVDet nên GPU 1 không bị chiếm bộ nhớ.

Sửa các biến trong cell **Configuration** để đổi lambda, epoch hoặc dataset
path. Branch Git phải chứa hỗ trợ `pixelwise_max`, `pseudo_aggregation=max` và
model không còn khóa cứng `cuda:0`; notebook sẽ kiểm tra và dừng với lỗi rõ ràng
nếu branch trên GitHub chưa được push.

Kết quả được lưu tại
`/kaggle/working/mvdet_pixelwise_max_2gpu_results.zip`, gồm log, checkpoint,
learning curve, manifest và tóm tắt pseudo-label generation.
