# Báo cáo phương pháp: SHOTBRL, MVDet cũ và YOLO pseudo labels

Ngày tổng hợp: 02/10/2026; bổ sung kết quả đối chứng do người dùng cung cấp. Phạm vi: mã nguồn trong repo và archive `C:\Users\ADMIN\Documents\mvdet_yolo26x_results.zip`. “SHOT” dưới đây là nhánh `SHOTBRL`; notebook YOLO26x huấn luyện nhánh `MVDetBRL`, không huấn luyện kiến trúc SHOT.

## Bảng so sánh phương pháp

| Tiêu chí | MVDet gốc / cấu hình cũ | SHOTBRL trong repo | MVDetBRL + YOLO26x trong notebook |
|---|---|---|---|
| Bài toán và đầu ra | Phát hiện vị trí người trên mặt phẳng đất (BEV) từ nhiều camera; thêm heatmap đầu/chân từng camera. | Cùng đầu ra BEV và heatmap từng camera. | Cùng đầu ra MVDet; YOLO chỉ cung cấp tín hiệu huấn luyện bổ sung. |
| Trích xuất và hợp nhất đặc trưng | Backbone ResNet18/VGG11; biến đổi phối cảnh đặc trưng mỗi camera về BEV bằng một phép chiếu; ghép các camera và dự đoán heatmap. | `DPerspTransDetector` mặc định: dự đoán phân bố độ sâu, chiếu đặc trưng qua nhiều mức sâu, tổng hợp rồi hợp nhất các camera. Mặc định `depth_scales=5`. | Giữ `PerspTransDetector` mặc định của MVDet với ResNet18; không dùng kiến trúc SHOT. |
| Nhãn và loss | Nhãn BEV và từng camera được làm mềm bằng Gaussian; tối ưu MSE. | Có lựa chọn MSE, BRL v1, BRL v2; mặc định BRL v2. BRL xử lý những vùng mô hình dự đoán cao nhưng không có nhãn như vùng có thể thiếu annotation. | BRL v1 trên nhãn thật, cộng loss pseudo từ YOLO26x trên BEV với trọng số `0.01`. |
| Vai trò YOLO | Không có. | Không có trong pipeline `SHOTBRL/main.py`. | YOLO26x phát hiện người từng ảnh camera, lưu bbox/confidence trong JSON offline. Điểm chân bbox được chiếu về BEV; notebook tạo Gaussian quanh vị trí đó, trọng số theo confidence. YOLO không chạy khi suy luận bằng checkpoint MVDet. |
| Thiếu annotation | MSE trên vùng thiếu nhãn có thể phạt một dự đoán người đúng như nền; đây là hệ quả của công thức loss. | BRL v1/v2 điều chỉnh loss ở vùng “confuse”; có tập giả lập bỏ 20%, 45%, 60% người trong các frame train. | Cùng mục tiêu BRL, thêm bằng chứng pseudo cho các vị trí bị thiếu nhãn; notebook dùng `drop_60`. |
| Đánh giá | Ngưỡng BEV, NMS rồi tính MODA, MODP, precision, recall. README gốc nêu 88,2% MODA trên Wildtrack đầy đủ nhãn. | Có cùng cách đánh giá; chưa thấy log benchmark SHOT đủ để kết luận cải thiện định lượng. | Archive ghi epoch 10: MODA **85,4%**, MODP **74,2%**, precision **93,4%**, recall **91,9%** trên Wildtrack với `drop_60`. |

## Quy trình training đã thực hiện trong notebook

| Bước | Thiết lập và tác dụng |
|---|---|
| 1. Chuẩn bị nhãn thiếu | Wildtrack, 90% frame train / 10% frame test. `drop_60` xóa khoảng 60% instance trong từng frame train; frame test giữ annotation đầy đủ. |
| 2. Tạo pseudo cache | YOLO26x, lớp `person` của COCO, confidence tối thiểu `0.20`, kích thước ảnh suy luận `1280`; lưu bbox và confidence theo frame/camera. |
| 3. Chiếu sang BEV | Lấy tâm cạnh dưới bbox làm điểm chân, dùng calibration camera để chiếu xuống mặt đất. Bỏ điểm ngoài BEV và vùng cách GT đã có dưới `0.5 m`. |
| 4. Tạo target pseudo | Gaussian với `sigma=0.5 m`; khi nhiều camera cùng dự đoán, lấy giá trị lớn nhất. Trọng số pixel là confidence YOLO nhân giá trị Gaussian. |
| 5. Tối ưu MVDet | ResNet18, BRL v1 cho nhãn thật + weighted MSE trên pseudo BEV. `batch_size=1`, 10 epochs, SGD (`lr=0.1`, momentum `0.5`, weight decay `5e-4`), OneCycleLR, seed `1`. |
| 6. Đánh giá | Test sau mỗi epoch bằng GT đầy đủ; heatmap ngưỡng `0.4`, NMS trước khi tính MODA/MODP. Checkpoint lưu sau mỗi epoch vào cùng đường dẫn; archive chứa checkpoint cuối epoch 10. |

## So sánh ba cấu hình tại `drop_60`

Kết quả BRL thuần và YOLO pseudo dạng `disk` do người dùng cung cấp; kết quả YOLO pseudo dạng `gaussian` lấy từ archive nêu trên. Chưa có log/manifest của hai run mới để đối chiếu encoder, seed, số epoch, checkpoint và ngưỡng đánh giá.

| Cấu hình | MODA (%) | MODP (%) | Precision (%) | Recall (%) |
|---|---:|---:|---:|---:|
| MVDet + BRL thuần | 81,5 | 74,5 | 96,4 | 84,7 |
| MVDet + BRL + YOLO pseudo `disk` | 83,1 | 74,3 | 92,7 | 90,2 |
| MVDet + BRL + YOLO pseudo `gaussian` | 85,4 | 74,2 | 93,4 | 91,9 |

So với BRL thuần, `disk` tăng MODA 1,6 điểm và recall 5,5 điểm, đồng thời precision giảm 3,7 điểm. So với `disk`, `gaussian` tăng MODA 2,3 điểm, precision 0,7 điểm và recall 1,7 điểm; MODP giảm 0,1 điểm. Cả ba cùng mức `drop_60`, nhưng chỉ có thể quy chênh lệch cho dạng pseudo target nếu những thiết lập còn lại cũng giống nhau.

## Cải tiến và mức độ chứng cứ

| Thay đổi | Lợi ích dự kiến / quan sát được | Kết luận hiện có |
|---|---|---|
| SHOT: chiếu nhiều mức sâu với trọng số học được | Có thể giảm sai số do giả định mọi đặc trưng nằm cùng mặt phẳng. | Đã có trong mã `SHOTBRL`; cần benchmark cùng split và loss để định lượng. |
| BRL v1/v2 | Giảm việc coi dự đoán ở vùng thiếu nhãn là false positive chắc chắn trong loss. V2 xác định positive từ điểm GT cứng, thay vì ngưỡng trên Gaussian của v1. | Đã có mã và script ablation; chưa có số đối chứng MSE/BRL cùng cấu hình trong archive đang xét. |
| YOLO pseudo labels offline | Bổ sung tín hiệu về người có thể bị thiếu annotation, không tăng phụ thuộc YOLO khi chạy MVDet inference. | Notebook và archive chứng minh pipeline chạy được; kết quả cuối là 85,4% MODA ở `drop_60`. |
| Gaussian pseudo target trong notebook | Target giảm dần theo khoảng cách từ điểm chân, có trọng số theo độ tin cậy; khác target đĩa phẳng có nhiễu trong `MVDetBRL/frameDataset.py` hiện tại. | Kết quả `disk` do người dùng cung cấp cho thấy `gaussian` cao hơn 2,3 điểm MODA ở `drop_60`; cần đối chiếu các thiết lập còn lại trước khi quy mức tăng riêng cho dạng target. |

**Lưu ý so sánh:** 88,2% MODA trong README MVDet gốc là con số tham khảo cho nhãn đầy đủ; 85,4% là kết quả lần chạy `drop_60` trong archive. Hai số không tạo thành phép so sánh hơn/kém trực tiếp. Log đánh giá YOLO cache tại bán kính ghép 0,5 m ghi precision/recall BEV sau NMS là **41,4% / 91,3%**; đây là chất lượng pseudo cache trên toàn bộ frame, không phải MODA của mô hình MVDet trên tập test. Các tỷ lệ precision/recall ở dòng `Train Epoch` là so khớp pixel thô với nhãn bị bỏ và cũng không thay thế chỉ số MODA. Các số BRL thuần và `disk` trong bảng là số liệu người dùng cung cấp, chưa được kiểm tra độc lập từ log.

## Nguồn kiểm tra

- [MVDet gốc và số tham khảo](MVDetBRL/README.md); [kiến trúc MVDet](MVDetBRL/multiview_detector/models/persp_trans_detector.py); [GaussianMSE](MVDetBRL/multiview_detector/loss/gaussian_mse.py).
- [Kiến trúc SHOT](SHOTBRL/multiview_detector/models/dpersp_trans_detector.py); [cấu hình train SHOT](SHOTBRL/main.py); [BRL v2](SHOTBRL/multiview_detector/loss/brl_gaussian_mse_v2.py).
- [Cấu hình MVDetBRL](MVDetBRL/main.py); [tạo pseudo cache](MVDetBRL/generate_pseudo_cache.py); [target pseudo trong repo](MVDetBRL/multiview_detector/datasets/frameDataset.py); [loss pseudo](MVDetBRL/multiview_detector/trainer.py); [notebook YOLO26x/Gaussian](mvdet-yolo26x-pseudo.ipynb).
- [Script giả lập thiếu annotation](simulate_dropped_annotations.py); [mã đánh giá](MVDetBRL/multiview_detector/evaluation/evaluate.py). Số liệu thực nghiệm lấy từ `run_manifest.json`, `gaussian_log/log.txt`, `yolo26x_pseudo_evaluation.log` trong archive nêu ở đầu báo cáo.
