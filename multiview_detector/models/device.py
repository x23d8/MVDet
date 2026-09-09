import torch


def resolve_model_devices():
    """Keep the original two-GPU split, with safe one-GPU/CPU fallbacks."""
    if not torch.cuda.is_available():
        cpu = torch.device('cpu')
        return cpu, cpu

    fusion_device = torch.device('cuda:0')
    front_device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else fusion_device
    return front_device, fusion_device
