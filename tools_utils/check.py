import torch, importlib.metadata
print(f"Torch  : {torch.__version__}  (CUDA {torch.version.cuda})")
print("flash-attn:", importlib.metadata.version("flash_attn"))
print("GPU     :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")