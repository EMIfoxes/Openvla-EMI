import torch

def check_torch_and_cuda():
    # 检测PyTorch版本
    print("PyTorch版本:", torch.__version__)
    
    # 检测CUDA是否可用
    if torch.cuda.is_available():
        print("CUDA可用")
        # 获取CUDA版本
        print("CUDA版本:", torch.version.cuda)
        # 获取可用的GPU数量
        print("可用的GPU数量:", torch.cuda.device_count())
        # 获取当前默认的GPU设备
        print("当前默认的GPU设备:", torch.cuda.current_device())
        # 获取当前默认GPU设备的名称
        print("当前默认GPU设备的名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA不可用")

# 调用函数
check_torch_and_cuda()