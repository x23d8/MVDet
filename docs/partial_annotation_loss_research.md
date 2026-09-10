# Hàm loss tổng quát cho MVDet với partial annotation

## Tóm tắt điều hành

MVDet gốc hồi quy một heatmap Gaussian bằng sai số Euclid ở cả mặt phẳng BEV và từng camera. Cách làm này phù hợp khi annotation đầy đủ, nhưng sai về mặt thống kê khi annotation bị thiếu: mọi vị trí không có điểm được ngầm coi là background, nên một pedestrian thật bị ẩn tạo gradient theo hướng ngược với pedestrian được quan sát. Đây chính là failure mode mà các nghiên cứu về missing-annotation detection, positive–unlabeled (PU) learning và incomplete dot annotation cùng chỉ ra.[^2][^3][^4][^6]

Thiết kế được chọn là **AdaptiveBRL**, một biến thể BRL có ràng buộc propensity và đồng thuận đa camera. Khác BRL gốc dùng activation của chính detector và một ngưỡng cố định, AdaptiveBRL:

- xem zero target là *unlabeled*, không mặc định là negative;
- suy ra số missing positive kỳ vọng từ xác suất annotation còn được giữ lại \(\rho\), thay vì đặt số pseudo-label cố định cho Wildtrack hoặc MultiviewX;
- chỉ khai thác local maxima có confidence cao hơn phân vị thấp của các positive đã quan sát trong cùng sample;
- dùng đồng thuận của hai camera tốt nhất sau phép chiếu hình học, rồi yêu cầu đồng thuận với BEV;
- giữ background focal loss nhưng giảm liên tục gradient ở nơi có bằng chứng foreground;
- warm-up và ramp pseudo supervision để hạn chế confirmation bias;
- dùng max-composed Gaussian để target luôn thuộc \([0,1]\), kể cả khi hai người đứng gần nhau.

Đây là phương trình có cơ sở tốt nhất trong phạm vi kiến trúc MVDet hiện tại và cơ chế drop ngẫu nhiên theo từng frame. Nó chưa phải bằng chứng rằng mọi ngưỡng metric sẽ chắc chắn đạt: kết luận đó chỉ hợp lệ sau thí nghiệm nhiều seed trên dữ liệu thật. Loss tác động mạnh đến precision/recall, nhưng MODP còn phụ thuộc phép chiếu, độ phân giải lưới và NMS.

## Bài toán và tiêu chí

MVDet nhận \(V\) ảnh đồng bộ, chiếu feature của từng camera lên ground plane và dự đoán occupancy heatmap. Bài báo gốc dùng Gaussian target và Euclidean loss:

\[
\mathcal L_{\mathrm{MVDet}}
=\lVert \tilde g-f(g)\rVert_2
+\frac{\alpha}{V}\sum_{v=1}^{V}
\left(\lVert\tilde s^v_h-f(s^v_h)\rVert_2+
\lVert\tilde s^v_f-f(s^v_f)\rVert_2\right).
\]

Thiết lập gốc đạt 88.2 MODA trên Wildtrack và 83.9 trên MultiviewX; hai dataset lần lượt có 7 và 6 camera, coverage trung bình 3.74 và 4.41 camera cho một vị trí.[^1] Điều này ủng hộ việc dùng tín hiệu camera agreement, nhưng cũng yêu cầu phép tổng hợp không phụ thuộc cứng vào số camera.

Các ngưỡng cần vượt, theo thứ tự MODA / MODP / Precision / Recall:

| Dataset và chế độ | MODA | MODP | Precision | Recall |
|---|---:|---:|---:|---:|
| Wildtrack, drop | 79.9 | 73.7 | 93.9 | 85.5 |
| Wildtrack, full/perfect | 88.0 | 74.7 | 93.2 | 95.0 |
| MultiviewX, drop | 66.5 | 78.2 | 98.5 | 67.5 |
| MultiviewX, full/perfect | 82.4 | 79.0 | 97.7 | 84.4 |

Các bộ số này không độc lập. Với \(P=TP/(TP+FP)\), \(R=TP/GT\), và định nghĩa MODA chuẩn bỏ qua chi tiết matching:

\[
\mathrm{MODA}=1-\frac{FN+FP}{GT}
=R\left(2-\frac{1}{P}\right).
\]

Ví dụ \(P=93.9\%\), \(R=85.5\%\) cho MODA xấp xỉ 79.9%. Vì vậy bài toán thực nghiệm thực chất là đạt đồng thời precision và recall; MODA sẽ đi theo hai đại lượng này. MODP là trục riêng về localization.

## Tổng hợp bằng chứng nghiên cứu

| Hướng | Kết quả chính | Giá trị cho MVDet | Hạn chế nếu dùng trực tiếp |
|---|---|---|---|
| Background Recalibration Loss | Đảo nhánh focal-negative thành nhánh positive khi activation rơi vào vùng “confusing”; trên COCO thiếu 50% label, BRL tăng mAP 0.237 lên 0.327.[^2] | Chứng minh hard negative trong missing-label setting thường là false negative và không nên nhận gradient negative lớn. | Dùng self-activation và threshold \(t\); bài báo ghi nhận threshold cao làm training không hội tụ sớm.[^2] |
| Soft Sampling | Giảm gradient của vùng background không chắc chắn nhưng vẫn giữ true-negative pool; bỏ toàn bộ vùng này kém hiệu quả.[^3] | Ủng hộ negative reweighting liên tục thay vì mask tất cả zero target. | Trọng số dựa IoU với box, không có tương đương tự nhiên cho point heatmap BEV. |
| AMSE cho incomplete dots | Asymmetric MSE cải thiện heatmap localization/counting ở drop 10–90%; hệ số tối ưu tăng theo drop rate.[^4] | Là bằng chứng trực tiếp rằng loss bất đối xứng hữu ích với Gaussian point target. | Một hệ số \(\beta\) phải tune theo dataset/model/drop rate, trái mục tiêu generalization. |
| nnPU risk | \(\pi R_p^+ + \max(0,R_u^- - \pi R_p^-)\) tránh negative empirical risk và overfit của unbiased PU.[^5] | Cho cách diễn giải đúng: observed points là positive, zero cells là unlabeled. | Pixel/grid-cell prior thay đổi theo grid size, Gaussian radius, ROI và visibility; dùng nnPU nguyên xi dễ tạo prior cực nhỏ và sai sampling assumptions. |
| PU object detection | PU loss cải thiện detector trên nhiều mức missingness; class prior động hữu ích hơn một prior cố định.[^6] | Ủng hộ dùng một đại lượng thích ứng theo sample/training thay vì hard-coded threshold. | Nghiên cứu áp dụng cho proposal objectness, không phải dense Gaussian occupancy grid. |
| Focal Loss | Giảm ảnh hưởng của số lượng rất lớn easy negatives trong dense detector.[^7] | Cần thiết vì mỗi pedestrian chỉ chiếm phần rất nhỏ của BEV. | Hard false negative lại chính là điểm có gradient lớn, nên focal đơn thuần làm missing-label failure nặng hơn. |
| PU dưới SCAR/SAR | Nếu labeled positives được chọn ngẫu nhiên từ positives, xác suất quan sát là một hệ số propensity; khi cơ chế label phụ thuộc đặc trưng (SAR), phải mô hình propensity theo instance.[^8][^9] | Simulator hiện tại drop ngẫu nhiên trong từng frame, nên \(\rho=1-d\) là thông tin hợp lệ. | Không được tuyên bố tổng quát cho annotation do người gán nhãn có thiên lệch occlusion/size nếu không ước lượng propensity. |
| Pseudo-positive mining | SparseDet cho thấy pseudo-label nhiễu gây suy giảm mạnh ở sparsity cao và tách labeled/unlabeled region tốt hơn naive pseudo-labeling.[^10] | Cần giới hạn số pseudo positive, confidence weighting và warm-up. | Box proposal/self-supervision khác heatmap đa camera. |
| Partial crowd annotation | Feature distribution và cross-regressor consistency cho phép dùng 10% annotated regions mà vẫn vượt các baseline semi-supervised/active learning.[^11] | Bằng chứng rằng consistency giữa các nguồn dự đoán là tín hiệu hữu ích cho vùng chưa gán nhãn. | Kiến trúc thay đổi lớn; không phải drop ngẫu nhiên từng instance trên toàn ảnh. |
| Teacher/consistency learning | Mean Teacher chỉ ra self-target có confirmation bias; weight averaging và ramp-up tăng độ ổn định.[^12] Soft Teacher dùng confidence làm trọng số pseudo-label.[^13] | Động cơ cho detach, confidence-squared weighting và ramp schedule. | Thêm EMA teacher sẽ gần gấp đôi state/memory của MVDet vốn đã nặng. |
| Adaptive pseudo thresholds | Consistent-Teacher cho thấy threshold tĩnh không theo kịp confidence giữa dataset và iteration; ngưỡng động ổn định pseudo targets.[^14] | Ủng hộ threshold tương đối theo observed positives thay cho score tuyệt đối. | GMM queue thêm state và chi phí; với một class và batch size 1, ước lượng dễ dao động. |
| Multi-view PU | Nguồn quan sát dị thể có thể cải thiện PU learning, nhưng reliable-negative identification sai sẽ làm giảm kết quả.[^15] | Camera là các view dị thể tự nhiên; chỉ nên tin vị trí có visibility và agreement. | Lý thuyết gốc không dùng projective geometry hay spatial heatmap. |
| Generalized MVD | Các mô hình MVD hiện tại overfit scene, vị trí/thứ tự và số camera; permutation invariance, regularization và thử nghiệm cross-scene là cần thiết.[^17] | Loss consensus phải không đổi khi thêm camera yếu và protocol không được chỉ kiểm tra một cấu hình camera. | Loss không thể tự làm kiến trúc concat feature của MVDet trở thành permutation-invariant. |
| Cardboard Human Modeling | MvCHM dùng focal loss cho occupancy và ghi nhận annotation bị thiếu nghiêm trọng gần biên vùng phát hiện của Wildtrack.[^18] | Xác nhận cả class imbalance lẫn label incompleteness tồn tại trong benchmark MVD thực. | Missing-at-border có cơ chế SAR khác simulator drop ngẫu nhiên, nên không thể chỉ dùng một global propensity để tuyên bố đã giải quyết hoàn toàn. |

Nghiên cứu mới về small-object localization cũng trực tiếp mô hình zero entries là tập unlabeled chứa cả positive lẫn background, dùng PU learning và loss modeling để đạt localization tốt với khoảng 10% điểm nhãn trên nhiều domain.[^16] Kết quả này củng cố lựa chọn PU/BRL cho MVDet, nhưng không thể chuyển thẳng con số vì benchmark, output representation và evaluation metric khác nhau.

## Phương trình AdaptiveBRL

### Target quan sát

Cho logit tại ô \(i\) là \(z_i\), xác suất \(p_i=\sigma(z_i)\), sparse point map là \(S\), và Gaussian kernel là \(K_G\). Target quan sát được tạo bằng max composition:

\[
G_i=\max_{j:S_j>0} S_j K_G(i-j), \qquad G_i\in[0,1].
\]

Phép max tránh trường hợp convolution cộng hai Gaussian gần nhau thành target lớn hơn 1, điều không hợp lệ với BCE probability.

### Bằng chứng đa camera

Foot probability của camera \(v\) được chiếu lên BEV thành \(a_{vi}\), kèm visibility \(m_{vi}\). Gọi \(a_{(1)i}\) và \(a_{(2)i}\) là hai score lớn nhất trong các camera nhìn thấy ô \(i\). Điểm đồng thuận là:

\[
c_i=\frac{a_{(1)i}+a_{(2)i}}{2}
\exp\left(-|a_{(1)i}-a_{(2)i}|\right).
\]

Top-two làm score không giảm chỉ vì dataset có thêm camera yếu/occluded. Thừa số agreement phạt trường hợp một camera rất tự tin nhưng camera thứ hai không xác nhận. Chỉ ô có ít nhất hai camera nhìn thấy mới dùng \(c_i\). Bằng chứng cuối cùng yêu cầu cả BEV và camera đồng ý:

\[
q_i=\sqrt{\operatorname{sg}(p_i)\operatorname{sg}(c_i)},
\]

trong đó \(\operatorname{sg}\) là stop-gradient. Nếu variant không có learned view head, \(q_i=\operatorname{sg}(p_i)\) và độ mạnh của kết luận giảm tương ứng.

### Budget từ annotation propensity

Simulator giữ độc lập xấp xỉ tỷ lệ \(\rho=1-d\) positive trong từng frame. Với \(n_o\) điểm quan sát, số điểm bị thiếu kỳ vọng là:

\[
K_{miss}=\operatorname{round}\left(\kappa n_o\frac{1-\rho}{\rho}\right),
\]

với \(\kappa=1\) mặc định. Công thức tự thích ứng với số người, dataset và drop rate. Nó không tạo pseudo-label nếu \(\rho=1\) hoặc frame không còn positive quan sát, tránh nghiệm suy diễn không kiểm soát.

Ứng viên phải đồng thời là local maximum của \(q\), nằm ngoài Gaussian halo đã quan sát, có đủ visibility, và thỏa:

\[
q_i>Q_{0.25}\left(\{q_j:S_j>0\}\right).
\]

Sau đó chỉ giữ top \(K_{miss}\). Phân vị được tính tương đối trong sample nên không gắn với calibration tuyệt đối của Wildtrack hay MultiviewX. Từ các điểm chọn được tạo pseudo Gaussian \(\hat G\) và confidence map \(w\) có peak \(q_i^2\).

### Ba thành phần loss

Đặt continuous focal BCE:

\[
\operatorname{FCE}(y,p)=|y-p|^\gamma
[-y\log p-(1-y)\log(1-p)].
\]

Ba mask là observed-positive \(\mathcal P=\{G_i>\tau_p\}\), pseudo-positive \(\mathcal M=\{\hat G_i>\tau_p\}\setminus\mathcal P\), và negative đáng tin \(\mathcal N\), tức các ô ngoài halo của cả hai target. Loss:

\[
\mathcal L_{obs}=\frac{1}{|\mathcal P|}\sum_{i\in\mathcal P}
\operatorname{FCE}(G_i,p_i),
\]

\[
\mathcal L_{pseudo}=\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M}
w_i\operatorname{FCE}(\hat G_i,p_i),
\]

\[
\mathcal L_{bg}=\frac{1}{|\mathcal N|}\sum_{i\in\mathcal N}
\left(1-r(t)q_i^2\right)p_i^\gamma[-\log(1-p_i)].
\]

\(r(t)\) bằng 0 trong warm-up, sau đó tăng tuyến tính tới 1. Đây là phần kế thừa tinh thần BRL: một hard negative có evidence foreground không nhận toàn bộ negative gradient. Khác BRL gốc, gradient không bị đảo chỉ do prediction của cùng một branch vượt threshold; nó được giảm mềm và chỉ top-budget candidates mới nhận positive gradient.

Loss BEV hoàn chỉnh:

\[
\boxed{
\mathcal L_{BEV}=\mathcal L_{obs}
+\lambda_{bg}\mathcal L_{bg}
+r(t)\lambda_{ps}\mathcal L_{pseudo}}
\]

và loss toàn mô hình:

\[
\boxed{
\mathcal L=\mathcal L_{BEV}
+\frac{\alpha}{V}\sum_{v=1}^{V}\mathcal L_{view}^{v}}
\]

Trong implementation mặc định: \(\gamma=2\), \(\lambda_{bg}=1\), \(\lambda_{ps}=0.25\), warm-up 1 epoch, ramp 3 epoch, \(\alpha=1\). Đây là các giá trị có ý nghĩa theo loss family, không có nhánh điều kiện theo tên dataset. View loss dùng cùng phương trình nhưng evidence là prediction đã detach của view đó; budget và adaptive reference vẫn ngăn pseudo-positive không giới hạn.

## Tại sao không chọn các phương án đơn giản hơn

**Gaussian MSE bất đối xứng đơn thuần.** AMSE rất sát loại output của MVDet, nhưng nghiên cứu gốc cho thấy \(\beta\) tối ưu thay đổi theo drop rate, dataset và model.[^4] Điều này không đáp ứng mục tiêu một cấu hình tổng quát.

**BRL nguyên bản.** BRL gốc hiệu quả và là nền tảng ý tưởng, nhưng threshold activation cao có thể làm training không ổn định ở giai đoạn đầu; chính tác giả quan sát hiện tượng không hội tụ.[^2] MVDet còn có tín hiệu camera độc lập mà RetinaNet một ảnh không có, nên bỏ qua tín hiệu này là lãng phí.

**Pure nnPU.** nnPU là lựa chọn lý thuyết sạch cho classification samples.[^5] Tuy nhiên một ô 10 cm, Gaussian support pixel và một proposal objectness không có cùng phân phối. Positive prior theo pixel còn thay đổi khi đổi `grid_reduce`, kernel hoặc ROI. AdaptiveBRL chỉ dùng propensity ở mức instance count, đúng với cách simulator loại instance, và dùng focal normalization riêng cho foreground/background.

**Naive pseudo-labeling.** Pseudo tất cả peak trên một threshold dễ khuếch đại false positives, đặc biệt ở drop cao; SparseDet và Consistent-Teacher đều nêu rủi ro pseudo-target nhiễu/dao động.[^10][^14] Budget, reference theo observed positives, confidence weighting và ramp là bốn hàng rào chống lỗi này.

**Bỏ toàn bộ zero target.** Khi không còn negative signal, heatmap có nghiệm foreground ở mọi nơi. Soft Sampling và BRL đều cho thấy giữ lại background có trọng số tốt hơn bỏ hoàn toàn.[^2][^3]

## Implementation trong repository

| Thành phần | File | Vai trò |
|---|---|---|
| AdaptiveBRL và multi-view consensus | `multiview_detector/loss/adaptive_brl.py` | Loss, pseudo budget, projection, visibility, agreement, thống kê |
| Gaussian target an toàn | `multiview_detector/loss/heatmap.py` | Max-composed Gaussian và masked mean |
| Trainer | `multiview_detector/trainer.py` | Truyền view/projection context, sigmoid khi output là logits, log loss statistics |
| CLI | `main.py` | `auto` chọn AdaptiveBRL khi `--pa > 0`; từ chối GaussianMSE không an toàn với partial labels |
| Dataset/evaluation | `frameDataset.py`, `path_utils.py` | Hỗ trợ drop rate bất kỳ; GT cache riêng train/test; evaluation dùng test GT |
| Unit tests | `tests/test_adaptive_brl.py` | Bound target, camera-count invariance, visibility, budget, warm-up, finite gradient, detach |

Việc tách train/test GT là bắt buộc để MODA/MODP đáng tin. Trước thay đổi, entrypoint truyền `train_set.gt_fpath` cho `test_loader`; nay dùng `test_set.gt_fpath`, và tên cache chứa split cùng fingerprint frame IDs.

## Protocol kiểm chứng bắt buộc

### Ma trận thí nghiệm

Chạy độc lập Wildtrack và MultiviewX với drop \(0,20,45,60\), ít nhất ba seed \(1,2,3\). Drop phải được tạo trước khi training, không dùng `hidden_annotations_positions` cho loss, early stopping, threshold hay chọn checkpoint.

| Nhóm | Loss | Mục đích |
|---|---|---|
| B0 | GaussianMSE | Baseline gốc; chỉ hợp lệ về mặt supervision ở drop 0 |
| B1 | Focal BCE trên zero-as-negative | Tách lợi ích do focal/class imbalance |
| B2 | BRL gốc với threshold | So sánh trực tiếp ý tưởng nguồn |
| B3 | AdaptiveBRL không camera consensus | Đo lợi ích propensity budget/adaptive threshold |
| B4 | AdaptiveBRL không budget | Đo confirmation-bias control |
| B5 | AdaptiveBRL đầy đủ | Phương án đề xuất |

Không tune một bộ hyperparameter riêng cho mỗi dataset. Chọn một cấu hình trên split validation được định trước, freeze nó, rồi chạy cả hai dataset. Báo cáo mean ± standard deviation và từng seed; không chỉ báo cáo seed tốt nhất.

### Quy tắc metric

- Giữ NMS radius 0.5 m như MVDet gốc.[^1]
- Dùng threshold 0.4 đã công bố trước cho bảng generalization chính.
- Nếu có threshold calibration, chỉ fit trên validation và báo thành bảng phụ; tuyệt đối không sweep trên test rồi chọn giá trị tốt nhất.
- Báo cáo đồng thời MODA, MODP, precision, recall và số detection/frame.
- Dùng full annotations chỉ ở validation/test evaluation, không trong training.
- Xác minh kết quả `pa=0` không giảm đáng kể so với GaussianMSE; chế độ `auto` hiện giữ nguyên GaussianMSE cho mục tiêu này.

### Lệnh chạy

```bash
python main.py -d wildtrack \
  --data_path /data/Wildtrack \
  --dropped_path /data/Wildtrack_dropped \
  --pa 45 --loss adaptive_brl --seed 1

python main.py -d multiviewx \
  --data_path /data/MultiviewX \
  --dropped_path /data/MultiviewX_dropped \
  --pa 45 --loss adaptive_brl --seed 1
```

Lặp tương tự cho drop/seed còn lại. Theo dõi `bev/pseudo_points`, `bev/expected_missing_points`, `bev/loss_*` và các thống kê view trên W&B. `pseudo_points` thấp kéo dài cho thấy camera consensus chưa đủ tốt; tăng pseudo weight không sửa được nguyên nhân này.

Mỗi run hoàn tất ghi `final_metrics.json`. Sau khi có ít nhất ba seed khác nhau cho từng cặp dataset/drop, kiểm tra đồng thời toàn bộ ngưỡng bằng:

```bash
python tools/check_metric_thresholds.py \
  logs/wildtrack_frame/default/pa45/*/final_metrics.json \
  logs/multiviewx_frame/default/pa45/*/final_metrics.json \
  --output logs/metric_gate_summary.json
```

Gate dùng điều kiện **lớn hơn nghiêm ngặt**, tách từng `pa`, từ chối seed trùng hoặc nhóm có dưới ba seed, và không chọn threshold trên test.

### Điều kiện chấp nhận

Một cấu hình chỉ được coi là đạt nếu **mean ba seed** vượt đồng thời bốn ngưỡng của bảng tương ứng. Ngoài ra:

- không seed nào bị collapse (MODA âm hoặc recall gần 0);
- chênh lệch giữa hai dataset không được giải quyết bằng dataset-specific constants;
- drop 60 không được dùng hidden labels để calibrate;
- full/perfect case dùng đúng full-label training path;
- log phải lưu commit, seed, drop statistics, threshold và resolved loss.

## Rủi ro và giới hạn kết luận

**Benchmark sơ bộ, chưa phải kết quả chấp nhận.** Môi trường hiện đã xác định được bản đầy đủ và các drop20/drop45/drop60 của cả Wildtrack lẫn MultiviewX trên ổ dữ liệu ngoài. Một run Wildtrack drop45, seed 1, một epoch đã hoàn tất trên RTX 4050 6 GB với AMP: MODA 32.46, MODP 73.47, precision 59.78 và recall 99.16. Epoch này nằm hoàn toàn trong warm-up (`ramp_factor=0`), nên chỉ chứng minh pipeline dữ liệu/đánh giá chạy end-to-end và chỉ ra failure mode over-detection; nó không chứng minh hiệu quả cuối cùng của pseudo supervision. Run 10 epoch tiếp theo được người dùng dừng tại batch 60 của epoch 1 và chưa sinh checkpoint theo cơ chế epoch-only cũ. Sau quan sát này, runtime được đổi sang checkpoint nguyên tử mặc định mỗi 25 batch và hỗ trợ resume đúng giữa epoch. Không metric mục tiêu nào được coi là đạt cho tới khi đủ ma trận nhiều seed và gate tự động pass.

Chi phí đo được dao động theo run: run hoàn tất mất khoảng 96 phút cho một epoch Wildtrack, còn run bị dừng có steady-state khoảng 19.2 giây/batch sau chi phí autotune đầu tiên khoảng 16 phút. Đánh giá test mất khoảng 6 phút. Vì vậy protocol dùng checkpoint đầy đủ theo batch và cho phép giảm số lần validation trung gian, nhưng luôn đánh giá epoch cuối. Đây là giới hạn thời gian thực nghiệm, không phải lý do để hạ điều kiện chấp nhận.

**Giả định SCAR.** Công thức budget đúng cho simulator drop ngẫu nhiên trong frame. Nếu annotation thực tế ưu tiên người lớn/rõ hoặc bỏ sót người occluded, đó là SAR; một \(\rho\) toàn cục sẽ bias. Khi đó nên học propensity \(\rho(x)\) theo visibility, projected size và occlusion, hoặc báo cáo riêng strata thay vì tuyên bố generalization.[^9]

**Calibration và precision–recall.** BCE logits tạo probability semantics tốt hơn raw MSE output, nhưng threshold 0.4 vẫn có thể chưa tối ưu. Không thể “đảm bảo” vừa tăng recall vừa giữ precision nếu không quan sát đường PR. MODA mục tiêu gần như được quyết định bởi cặp precision/recall đã nêu; MODP sẽ không tự tăng chỉ nhờ missing-label loss.

**View pseudo-labels.** View branch không có EMA teacher để tiết kiệm bộ nhớ. Detach, budget và ramp giảm self-confirmation nhưng không loại bỏ hoàn toàn. Nếu B5 còn dao động giữa seed, nâng cấp ưu tiên là EMA teacher chỉ cho hai classifier heads hoặc temporal ensembling cache, không phải thêm threshold dataset-specific.[^12][^13]

**Variant `img_proj`.** Variant này không có learned per-view detector, nên không thể dùng cross-view foot consensus và rơi về self-evidence. Kỳ vọng generalization mạnh nhất áp dụng cho `default`, là kiến trúc MVDet chuẩn.

**Generalization của toàn hệ thống.** Top-two consensus làm riêng loss không phụ thuộc số camera, nhưng backbone MVDet vẫn concat feature theo camera và vì thế chưa permutation-invariant. Nghiên cứu GMVD cho thấy đây là giới hạn kiến trúc thực sự khi đổi scene/cấu hình camera.[^17] Kết luận “generalize” của nhánh này do đó chỉ nên hiểu là cùng một phương trình và hyperparameter cho Wildtrack/MultiviewX cùng nhiều drop rate; cross-scene/camera generalization cần một thay đổi kiến trúc riêng và benchmark GMVD.

## Kết luận

AdaptiveBRL giải đúng failure mode của partial annotation hơn GaussianMSE và cải thiện điểm yếu chính của BRL gốc bằng ba nguồn ràng buộc: annotation propensity, geometry-aware multi-view agreement và adaptive confidence relative to observed positives. Phương trình không chứa tên dataset, không giả định 6 hay 7 camera, và nhận drop rate bất kỳ. Thiết kế này là ứng viên hợp lý nhất để đưa vào benchmark; tuy nhiên các ngưỡng Wildtrack/MultiViewX chỉ được xác nhận sau ma trận thí nghiệm nhiều seed nói trên, không thể được đảm bảo chỉ từ suy luận hay unit test.

## Sources

[^1]: Yunzhong Hou, Liang Zheng, Stephen Gould. “[Multiview Detection with Feature Perspective Transformation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520001.pdf).” ECCV, 2020.
[^2]: Han Zhang et al. “[Solving Missing-Annotation Object Detection with Background Recalibration Loss](https://arxiv.org/html/2002.05274).” ICASSP, 2020.
[^3]: Zhe Wu et al. “[Soft Sampling for Robust Object Detection](https://arxiv.org/abs/1806.06986).” BMVC, 2018.
[^4]: Feng Chen, Michael P. Pound, Andrew P. French. “[Learning to Localise and Count With Incomplete Dot-Annotations](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/html/Chen_Learning_to_Localise_and_Count_With_Incomplete_Dot-Annotations_ICCVW_2021_paper.html).” ICCV Workshops, 2021.
[^5]: Ryuichi Kiryo et al. “[Positive-Unlabeled Learning with Non-Negative Risk Estimator](https://proceedings.neurips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html).” NeurIPS, 2017.
[^6]: Yuewei Yang, Kevin J. Liang, Lawrence Carin. “[Object Detection as a Positive-Unlabeled Problem](https://arxiv.org/html/2002.04672).” BMVC, 2020.
[^7]: Tsung-Yi Lin et al. “[Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html).” ICCV, 2017.
[^8]: Charles Elkan, Keith Noto. “[Learning Classifiers from Only Positive and Unlabeled Data](https://cseweb.ucsd.edu/~elkan/posonly.pdf).” KDD, 2008.
[^9]: Jessa Bekker, Jesse Davis. “[Learning from Positive and Unlabeled Data under the Selected At Random Assumption](https://proceedings.mlr.press/v94/bekker18a.html).” PMLR 94, 2018.
[^10]: Saksham Suri et al. “[SparseDet: Improving Sparsely Annotated Object Detection with Pseudo-positive Mining](https://arxiv.org/abs/2201.04620).” ICCV, 2023.
[^11]: Yanyu Xu et al. “[Crowd Counting With Partial Annotations in an Image](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_Crowd_Counting_With_Partial_Annotations_in_an_Image_ICCV_2021_paper.html).” ICCV, 2021.
[^12]: Antti Tarvainen, Harri Valpola. “[Mean Teachers Are Better Role Models](https://arxiv.org/abs/1703.01780).” NeurIPS, 2017.
[^13]: Mengde Xu et al. “[End-to-End Semi-Supervised Object Detection with Soft Teacher](https://arxiv.org/abs/2106.09018).” ICCV, 2021.
[^14]: Xinjiang Wang et al. “[Consistent-Teacher: Towards Reducing Inconsistent Pseudo-Targets in Semi-Supervised Object Detection](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Consistent-Teacher_Towards_Reducing_Inconsistent_Pseudo-Targets_in_Semi-Supervised_Object_Detection_CVPR_2023_paper.html).” CVPR, 2023.
[^15]: Joey Tianyi Zhou et al. “[Multi-view Positive and Unlabeled Learning](https://proceedings.mlr.press/v25/zhou12.html).” ACML, 2012.
[^16]: Xiao Zhou et al. “[Small Object Localization with 90% Annotation Reduction by Positive-Unlabeled Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC12735266/).” Micromachines, 2025.
[^17]: Jeet Vora et al. “[Bringing Generalization to Deep Multi-View Pedestrian Detection](https://openaccess.thecvf.com/content/WACV2023W/RWS/html/Vora_Bringing_Generalization_to_Deep_Multi-View_Pedestrian_Detection_WACVW_2023_paper.html).” WACV Workshops, 2023.
[^18]: Jiahao Ma et al. “[Multiview Detection with Cardboard Human Modeling](https://arxiv.org/abs/2207.02013).” ACCV, 2024.
