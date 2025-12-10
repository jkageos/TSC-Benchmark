import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(
    f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}"
)
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
