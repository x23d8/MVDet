"""Point-BRL cho bản đồ BEV với nhãn người đi bộ dạng điểm."""

import torch
import torch.nn.functional as F
from torch import nn


class PointBRLLoss(nn.Module):
    """Chia BEV thành điểm thật, âm dễ và âm đáng ngờ theo xác suất dự đoán.

    Đầu vào là logits [B, 1, H, W] và bản đồ điểm GT cùng batch/kênh.
    Nếu độ phân giải GT khác, max pooling giữ lại mọi điểm đã annotate.
    ``epoch`` bắt đầu từ 0; trong ``warmup_epochs`` epoch đầu, mọi ô
    không có GT đều nhận gradient âm thông thường.
    """

    def __init__(self, confusion_threshold=0.7, gamma=2.0,
                 positive_weight=1.0, confusion_weight=0.1,
                 background_weight=1.0, warmup_epochs=2,
                 guard_radius=0, enable_confusion=True):
        super().__init__()
        if not 0.0 < confusion_threshold < 1.0:
            raise ValueError('confusion_threshold must be strictly between 0 and 1')
        if gamma < 0 or min(positive_weight, confusion_weight, background_weight) < 0:
            raise ValueError('gamma and loss weights must be non-negative')
        if warmup_epochs < 0 or guard_radius < 0:
            raise ValueError('warmup_epochs and guard_radius must be non-negative')

        self.confusion_threshold = confusion_threshold
        self.gamma = gamma
        self.positive_weight = positive_weight
        self.confusion_weight = confusion_weight
        self.background_weight = background_weight
        self.warmup_epochs = warmup_epochs
        self.guard_radius = guard_radius
        self.enable_confusion = enable_confusion
        self.epoch = 0

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError('epoch must be non-negative')
        self.epoch = epoch

    @staticmethod
    def _group_mean(values, mask):
        # Nhóm rỗng đóng góp 0 nhưng vẫn giữ loss nối với đồ thị autograd.
        return (values * mask).sum() / mask.sum().clamp_min(1)

    def point_mask(self, logits, target):
        if logits.ndim != 4 or target.ndim != 4 or logits.shape[1] != 1 or target.shape[1] != 1:
            raise ValueError('logits and target must have shape [B, 1, H, W]')
        if logits.shape[0] != target.shape[0]:
            raise ValueError('logits and target must have the same batch size')
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.adaptive_max_pool2d(target, logits.shape[-2:])
        return target > 0

    def forward(self, logits, target):
        positive = self.point_mask(logits, target)
        non_gt = ~positive

        # Guard chỉ bảo vệ lân cận GT đã biết khỏi nhánh mirror; không dùng
        # khoảng cách để suy đoán một ô chưa gán nhãn có người hay không.
        guard = positive
        if self.guard_radius:
            width = 2 * self.guard_radius + 1
            guard = F.max_pool2d(positive.float(), width, stride=1,
                                 padding=self.guard_radius) > 0

        # Quyết định phân nhóm không được truyền gradient ngược qua confidence.
        # Validation dùng annotation đầy đủ: không mirror false positive.
        if self.training and self.enable_confusion and self.epoch >= self.warmup_epochs:
            confused = non_gt & ~guard & (logits.detach().sigmoid() > self.confusion_threshold)
        else:
            confused = torch.zeros_like(positive)
        background = non_gt & ~confused

        # softplus(z) = -log(1-sigmoid(z)); softplus(-z) = -log(sigmoid(z)).
        # Tính trên logits tránh log(0) khi model rất tự tin.
        probability = logits.sigmoid()
        positive_loss = (1 - probability).pow(self.gamma) * F.softplus(-logits)
        background_loss = probability.pow(self.gamma) * F.softplus(logits)
        # Nhóm đáng ngờ vẫn mang nhãn âm về ngữ nghĩa, nhưng dùng nhánh
        # positive được phản chiếu để giảm phạt một người bị thiếu annotation.
        confused_loss = positive_loss

        # Lấy trung bình từng nhóm để số ô nền không lấn át các điểm GT.
        return (self.positive_weight * self._group_mean(positive_loss, positive)
                + self.confusion_weight * self._group_mean(confused_loss, confused)
                + self.background_weight * self._group_mean(background_loss, background))
