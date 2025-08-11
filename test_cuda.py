import torch
print("torch:", torch.__version__)
print("torch.cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    a = torch.randn(1024,1024, device="cuda")
    b = torch.randn(1024,1024, device="cuda")
    c = a@b
    torch.cuda.synchronize()
    print("matmul OK, shape:", c.shape)