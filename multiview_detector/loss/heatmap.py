import torch
import torch.nn.functional as F


def _kernel_for_channel(kernel, channel):
    if kernel.ndim == 2:
        return kernel
    if kernel.ndim != 4:
        raise ValueError('kernel must have shape [H, W] or [C, C, H, W]')
    return kernel[min(channel, kernel.shape[0] - 1), min(channel, kernel.shape[1] - 1)]


def max_gaussian_target(prediction, sparse_target, kernel):
    """Expand sparse (possibly weighted) points using max-composed Gaussians.

    Max composition is important for a probability target: convolution would
    make overlapping people produce values above one. Point values are kept,
    which also lets the partial-label loss construct confidence maps.
    """
    if sparse_target.ndim == 3:
        sparse_target = sparse_target.unsqueeze(1)
    if prediction.ndim != 4 or sparse_target.ndim != 4:
        raise ValueError('prediction and target must have shape [B, C, H, W]')

    # Target construction has no gradient. Stamping dozens of tiny patches on
    # CUDA launches one kernel per annotated person and is dramatically slower
    # than doing the sparse work on CPU, then transferring one dense heatmap.
    work_dtype = torch.float32
    pooled = F.adaptive_max_pool2d(
        sparse_target.detach().to(device='cpu', dtype=work_dtype),
        prediction.shape[-2:],
    )
    if pooled.shape[1] != prediction.shape[1]:
        if pooled.shape[1] == 1:
            pooled = pooled.expand(-1, prediction.shape[1], -1, -1)
        else:
            raise ValueError('target channels do not match prediction channels')

    target = torch.zeros(prediction.shape, device='cpu', dtype=work_dtype)
    kernel = kernel.detach().to(device='cpu', dtype=work_dtype)
    height, width = prediction.shape[-2:]
    with torch.no_grad():
        for batch_idx in range(prediction.shape[0]):
            for channel_idx in range(prediction.shape[1]):
                gaussian = _kernel_for_channel(kernel, channel_idx)
                kernel_h, kernel_w = gaussian.shape
                center_y, center_x = kernel_h // 2, kernel_w // 2
                points = torch.nonzero(pooled[batch_idx, channel_idx] > 0, as_tuple=False)
                for point_y, point_x in points.tolist():
                    out_y0, out_x0 = max(point_y - center_y, 0), max(point_x - center_x, 0)
                    out_y1 = min(point_y - center_y + kernel_h, height)
                    out_x1 = min(point_x - center_x + kernel_w, width)
                    ker_y0, ker_x0 = out_y0 - point_y + center_y, out_x0 - point_x + center_x
                    ker_y1, ker_x1 = ker_y0 + out_y1 - out_y0, ker_x0 + out_x1 - out_x0
                    value = pooled[batch_idx, channel_idx, point_y, point_x]
                    patch = gaussian[ker_y0:ker_y1, ker_x0:ker_x1] * value
                    current = target[batch_idx, channel_idx, out_y0:out_y1, out_x0:out_x1]
                    target[batch_idx, channel_idx, out_y0:out_y1, out_x0:out_x1] = torch.maximum(
                        current, patch
                    )
    return target.clamp_(0.0, 1.0).to(device=prediction.device, dtype=prediction.dtype)


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)
