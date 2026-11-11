import torch
print(torch.cuda.is_available())         # powinno zwrócić True
print(torch.version.cuda)                # Powinno być: 11.8
print(torch.cuda.get_device_name(0))     # np. NVIDIA RTX 4050
