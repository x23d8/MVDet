# Point-based BEV loss: pseudo code gốc

Nguồn: nhánh `debrl`, commit `007c273`, file `spec.md`, mục **Pseudocode chốt**.
Mã triển khai của thiết kế này ở `debrl:multiview_detector/loss/point_brl.py`.

```text
Input:
    logits Z
    point GT mask Y
    confusion threshold c
    gamma
    lambda_conf
    warmup_epochs

P = sigmoid(Z)

POS = (Y == 1)
NON_GT = not POS

# Optional: guard nhỏ quanh point GT đã biết, chỉ để tránh đào lại
# các pixel sát positive thành confused negative.
GUARD = build_small_gt_guard(POS)

if epoch < warmup_epochs:
    CONF = false
    BG = NON_GT
else:
    # Prediction chỉ quyết định phân nhóm; không truyền gradient qua bước này.
    P_route = stop_gradient(P)
    CONF = NON_GT AND not GUARD AND (P_route > c)
    BG = NON_GT AND not CONF

L_pos(i)  = -(1 - P[i])^gamma * log(P[i])
L_bg(i)   = -P[i]^gamma * log(1 - P[i])

# Nhánh mirror cho điểm âm đáng ngờ: pt = 1 - P[i].
pt = 1 - P[i]
L_conf(i) = -pt^gamma * log(1 - pt)

L = mean(L_pos over POS)
  + lambda_conf * mean(L_conf over CONF)
  + mean(L_bg over BG)
```

`POS` là các point được gán nhãn, `CONF` là những ô không có nhãn nhưng mô
hình dự đoán đủ tự tin, và `BG` là phần nền còn lại. Trong warm-up, toàn bộ
ô không có GT đi qua nhánh âm thông thường. Khi đánh giá với nhãn đầy đủ,
triển khai gốc tắt nhánh `CONF`.

**Trạng thái trong `cfgmsel`:** tài liệu này giữ lại thiết kế Point-BRL của
bạn. Các lệnh `--loss confuse_gaussian` trong nhánh này vẫn chạy
ConfuseGaussianMSE; chúng chưa chạy PointBRLLoss. Hai loss có cách tạo target,
phân nhóm và công thức khác nhau.
