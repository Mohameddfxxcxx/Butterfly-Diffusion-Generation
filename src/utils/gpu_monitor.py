"""GPU memory and device info helpers."""
import torch


def gpu_mem_mb() -> float:
    return torch.cuda.memory_allocated() / 1e6


def gpu_info() -> dict:
    assert torch.cuda.is_available(), "CUDA GPU required."
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_vram_gb": round(props.total_memory / 1e9, 2),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "capability": f"{props.major}.{props.minor}",
    }
