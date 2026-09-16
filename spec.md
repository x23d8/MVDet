Sau khi đối chiếu lại **BRL gốc**, đặc tính **anchor-free/BEV của MVDet**, và idea 3 vùng của bạn, mình chốt phương án cuối là:

$$
\boxed{\textbf{Confidence-Guided Point-BRL}}
$$

**Không dùng feature distance. Không dùng Euclidean distance để quyết định một điểm có phải missing pedestrian hay không.** Spatial distance chỉ nên dùng như một **vùng bảo vệ nhỏ quanh GT đã biết** nếu cần tránh nhầm phần lân cận GT với confused point.

BRL gốc cần IoU vì nó làm việc trên anchor: IoU thấp được dùng để cô lập một pool các “confusion anchors”, rồi chính activation \(p_t\) mới quyết định có mirror gradient hay không. Paper cũng giải thích rằng nếu feature của một hard negative giống object thật thì classifier tự sinh ra \(p_t\) thấp; tức confidence đã đóng vai trò proxy cho feature resemblance. Vì vậy thêm một feature-distance criterion riêng là khá dư thừa và tạo thêm hyperparameter. ([arXiv][1])

## Design cuối

MVDet dự đoán một BEV pedestrian map; bản thân MVDet là anchor-free và suy luận pedestrian location từ ground-plane feature map, nên không cần tạo bbox/anchor giả chỉ để lấy IoU. ([arXiv][2])

Với BEV cell \(i\):

$$
p_i=\sigma(z_i)=P(\text{pedestrian tại }i)
$$

và binary point GT:

$$
y_i=
\begin{cases}
1 & \text{annotated pedestrian}\\
0 & \text{không có annotation}
\end{cases}
$$

Chia thành đúng **3 nhóm**:

$$
\boxed{
r_i=
\begin{cases}
\text{Positive},&
y_i=1
\\
\text{Confused Negative},&
y_i=0,\; i\notin G_{\text{guard}},\; p_i>c
\\
\text{Normal Background},&
\text{còn lại}
\end{cases}}
$$

Trong đó \(G_{\text{guard}}\) là một vùng rất nhỏ quanh GT đã annotate. Nó **không dùng để tìm missing pedestrian**, chỉ để tránh một peak thuộc GT thật bị phần lân cận của nó khai thác lại thành confused negative.

Nếu target của bạn chỉ đúng **một point**, guard có thể chỉ là point đó hoặc radius rất nhỏ. Nếu vẫn giữ Gaussian heatmap của MVDet thì guard nên bao phủ vùng Gaussian-positive thay vì coi các pixel lân cận là background.

---

### 1. Positive

Annotated GT chắc chắn là foreground:

$$
\boxed{
L_i^{pos}
=
-\alpha_{pos}(1-p_i)^\gamma
\log(p_i+\epsilon)
}
$$

Mục tiêu:

$$
p_i\rightarrow1
$$

---

### 2. Normal Background

Không có GT và model cũng không đủ tự tin rằng có người:

$$
p_i\le c
$$

Dùng negative Focal Loss:

$$
\boxed{
L_i^{bg}
=
-\alpha_{bg}p_i^\gamma
\log(1-p_i+\epsilon)
}
$$

Mục tiêu:

$$
p_i\rightarrow0
$$

---

### 3. Confused Negative

Đây là trường hợp quan trọng:

$$
y_i=0
$$

nhưng:

$$
p_i>c
$$

Annotation nói background nhưng model rất tin pedestrian.

Với negative:

$$
p_{t,i}=1-p_i
$$

Dùng mirrored BRL:

$$
\boxed{
L_i^{conf}
=
-\alpha_{conf}
p_{t,i}^{\gamma}
\log(1-p_{t,i}+\epsilon)
}
$$

hay tương đương:

$$
\boxed{
L_i^{conf}
=
-\alpha_{conf}
(1-p_i)^\gamma
\log(p_i+\epsilon)
}
$$

Đây là đúng cơ chế BRL: negative branch của Focal Loss được thay bằng mirrored positive branch khi negative activation xuống dưới confusion threshold. ([arXiv][1])

Nhưng **không đổi \(y_i\) thành 1**. Về semantic, nó vẫn là *suspicious negative*, không phải pseudo-label.

---

## Quan hệ giữa \(c\) của bạn và \(t\) của BRL

BRL gốc xét:

$$
p_t<t
$$

Với negative:

$$
p_t=1-p
$$

nên:

$$
1-p<t
$$

$$
p>1-t
$$

Do đó:

$$
\boxed{c=1-t}
$$

Ví dụ:

$$
t=0.3
$$

thì:

$$
\boxed{c=0.7}
$$

Tức model phải dự đoán pedestrian trên 70% mới vào confused branch.

---

## Loss tổng

Mình khuyên **average riêng từng nhóm**, không average trực tiếp toàn bộ grid vì background áp đảo:

$$
L_{pos}
=
\frac{1}{N_{pos}}\sum_{i\in P}L_i^{pos}
$$

$$
L_{conf}
=
\frac{1}{N_{conf}}\sum_{i\in C}L_i^{conf}
$$

$$
L_{bg}
=
\frac{1}{N_{bg}}\sum_{i\in B}L_i^{bg}
$$

Cuối cùng:

$$
\boxed{
L_{\text{Point-BRL}}
=
\lambda_{pos}L_{pos}
+
\lambda_{conf}L_{conf}
+
\lambda_{bg}L_{bg}
}
$$

Mình sẽ bắt đầu với:

$$
\lambda_{pos}=1,\qquad
\lambda_{bg}=1
$$

còn:

$$
\lambda_{conf}<1
$$

vì chính confused pool chứa hỗn hợp:

$$
\text{missing pedestrian}
+
\text{false positive thật}
$$

BRL gốc cũng cho thấy scaling contribution của confusion anchors rất quan trọng; trong thí nghiệm extreme missing-label của họ, weight 0.1 cho confusion anchors đạt kết quả tốt nhất trong các weight được thử, nhưng đây **không phải giá trị mặc định chắc chắn tối ưu cho MVDet**. ([arXiv][1])

Mình sẽ ablate:

$$
\lambda_{conf}\in\{0.1,0.25,0.5\}
$$

---

## Có một thứ mình chắc chắn thêm: warm-up

Đầu training:

$$
p_i
$$

chưa đáng tin. Nếu ngay epoch 1 cứ:

$$
p_i>c\Rightarrow mirror
$$

thì prediction sai ngẫu nhiên có thể tự được bảo vệ.

BRL paper cũng nói classifier cần đủ trưởng thành để có thể tin activation của chính model, và threshold quá mạnh có thể gây gradient không ổn định ở giai đoạn sớm. ([arXiv][1])

Do đó:

$$
e<E_{warm}
$$

→ **không có confused branch**, tất cả non-GT dùng normal negative loss.

Sau warm-up:

$$
e\ge E_{warm}
$$

→ bật Point-BRL.

Ví dụ với training 10 epoch, mình sẽ thử:

$$
E_{warm}=2
$$

hoặc 3 epoch và ablate.

---

## Pseudocode chốt

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

# optional:
# small region around annotated GT;
# only prevents re-mining known positives
GUARD = build_small_gt_guard(POS)

if epoch < warmup_epochs:

    CONF = false
    BG   = NON_GT

else:

    # detach because prediction is used only for routing
    P_route = stop_gradient(P)

    CONF =
        NON_GT
        AND not GUARD
        AND (P_route > c)

    BG =
        NON_GT
        AND not CONF


# -----------------------------
# Positive
# -----------------------------
L_pos(i) =
    -(1 - P[i])^gamma
     log(P[i])


# -----------------------------
# Normal background
# -----------------------------
L_bg(i) =
    -P[i]^gamma
     log(1 - P[i])


# -----------------------------
# Confused negative
# -----------------------------
pt = 1 - P[i]

L_conf(i) =
    -pt^gamma
     log(1 - pt)


L =
      mean(L_pos over POS)
    + lambda_conf * mean(L_conf over CONF)
    + mean(L_bg over BG)

return L
```

### Vậy distance và feature distance cuối cùng nằm đâu?

Mình chốt như sau:

| Thành phần                               | Final design             |
| ---------------------------------------- | ------------------------ |
| IoU                                      | ❌ Không dùng             |
| Euclidean distance để tìm missing person | ❌ Không dùng             |
| Euclidean distance làm GT guard          | ✅ Có thể dùng, rất nhỏ   |
| Feature Euclidean distance               | ❌ Không dùng             |
| Cosine feature similarity                | ❌ Không dùng             |
| Model confidence \(p_i\)                 | ✅ Tín hiệu chính         |
| Threshold \(c\)                          | ✅ Chọn confused negative |
| Mirrored BRL                             | ✅                        |
| Warm-up                                  | ✅                        |
| Confusion loss weight                    | ✅                        |

Lý do mình **không chọn spatial distance làm criterion chính** là một missing pedestrian hoàn toàn có thể đứng xa **hoặc rất gần** một annotated pedestrian. Distance tới annotated GT không cung cấp bằng chứng chắc chắn nó là người hay background.

Lý do mình **không chọn feature distance** là BRL vốn đã dựa vào classifier activation như một biểu hiện của feature resemblance; thêm một metric embedding nữa vừa trùng tín hiệu vừa buộc bạn phải định nghĩa prototype/person feature, normalization, metric và threshold mới. ([arXiv][1])

Vì vậy contribution của bạn sẽ rất sạch:

$$
\boxed{
\text{Anchor-based BRL}
\longrightarrow
\text{Point-based confidence-guided BRL on BEV}
}
$$

với điểm mới nằm ở **cách mining suspicious negatives trực tiếp trên BEV heatmap**, thay vì IoU-based anchor mining. Đây là design mình sẽ dùng làm **phương pháp chính**; distance/feature-similarity chỉ nên để làm ablation extension, không cho vào core method.

[1]: https://arxiv.org/abs/2002.05274 "Solving Missing-Annotation Object Detection with Background Recalibration Loss"
[2]: https://arxiv.org/abs/2007.07247?utm_source=chatgpt.com "Multiview Detection with Feature Perspective Transformation"
