import platform
import psutil
import torch

def bytes_to_gb(bytes_val):
    return bytes_val / 1024 ** 3

print("="*40)
print("🔍 训练环境检测")
print("="*40)

# 系统信息
print(f"操作系统: {platform.system()} {platform.release()} ({platform.version()})")
print(f"Python 版本: {platform.python_version()}")

# CPU 信息
cpu_name = platform.processor() or "未知CPU"
print(f"CPU 型号: {cpu_name}")
print(f"CPU 核心数: {psutil.cpu_count(logical=False)} 物理 / {psutil.cpu_count(logical=True)} 逻辑")

# 内存信息
mem = psutil.virtual_memory()
print(f"内存总量: {bytes_to_gb(mem.total):.2f} GB")

# GPU 信息
if torch.cuda.is_available():
    print(f"检测到 {torch.cuda.device_count()} 块 GPU")
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {prop.name}, 显存: {bytes_to_gb(prop.total_memory):.2f} GB")
    print(f"CUDA 版本: {torch.version.cuda}")
else:
    print("⚠ 未检测到可用 GPU（CUDA 不可用）")

print("="*40)