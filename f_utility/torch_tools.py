import torch

def try_gpu():
    """尝试使用 GPU（CUDA），如果不可用，则回退到 CPU"""
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ 发现 GPU，使用 CUDA 加速！🚀")
    else:
        device = "cpu"
        print("⚠️ 未发现 GPU，使用 CPU 运行，可能会较慢。")

    return device
