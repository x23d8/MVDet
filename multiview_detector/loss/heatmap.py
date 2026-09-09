import torch
import torch.nn.functional as F


def _kernel_for_channel(kernel, channel):
    if kernel.ndim == 2:
        return kernel
    if kernel.ndim != 4:
        raise ValueError('Gaussian kernel must have shape [H, W] or [C, C, H, W]')
    output_channel = min(channel, kernel.shape[0] - 1)
    input_channel = min(channel, kernel.shape[1] - 1)
    return kernel[output_channel, input_channel]


def max_gaussian_target(prediction, sparse_target, kernel):
    """Expand sparse points with Gaussian kernels without summing overlaps.

    MVDet's historical target transform used convolution, so two nearby people
    could produce targets larger than one.  A pixel belongs to the closest /
    strongest point target, therefore max-composition is the appropriate
    operation for both regression and probability losses.
    """
    if sparse_target.ndim == 3:
        sparse_target = sparse_target.unsqueeze(1)
    if prediction.ndim != 4 or sparse_target.ndim != 4:
        raise ValueError('prediction and sparse_target must have shape [B, C, H, W]')

    sparse_target = sparse_target.to(device=prediction.device)
    pooled = F.adaptive_max_pool2d(sparse_target.float(), prediction.shape[-2:])
    if pooled.shape[1] != prediction.shape[1]:
        if pooled.shape[1] == 1:
            pooled = pooled.expand(-1, prediction.shape[1], -1, -1)
        else:
            raise ValueError('Target channels do not match prediction channels')

    target = torch.zeros_like(prediction)
    kernel = kernel.to(device=prediction.device, dtype=prediction.dtype)
    height, width = prediction.shape[-2:]

    with torch.no_grad():
        for batch_idx in range(prediction.shape[0]):
            for channel_idx in range(prediction.shape[1]):
                gaussian = _kernel_for_channel(kernel, channel_idx)
                kernel_h, kernel_w = gaussian.shape[-2:]
                center_y, center_x = kernel_h // 2, kernel_w // 2
                points = torch.nonzero(pooled[batch_idx, channel_idx] > 0, as_tuple=False)
                for point_y, point_x in points.tolist():
                    out_y0 = max(point_y - center_y, 0)
                    out_x0 = max(point_x - center_x, 0)
                    out_y1 = min(point_y - center_y + kernel_h, height)
                    out_x1 = min(point_x - center_x + kernel_w, width)

                    kernel_y0 = out_y0 - (point_y - center_y)
                    kernel_x0 = out_x0 - (point_x - center_x)
                    kernel_y1 = kernel_y0 + out_y1 - out_y0
                    kernel_x1 = kernel_x0 + out_x1 - out_x0

                    current = target[batch_idx, channel_idx, out_y0:out_y1, out_x0:out_x1]
                    patch = gaussian[kernel_y0:kernel_y1, kernel_x0:kernel_x1]
                    target[batch_idx, channel_idx, out_y0:out_y1, out_x0:out_x1] = torch.maximum(
                        current,
                        patch,
                    )
    return target.clamp_(0.0, 1.0)


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)
