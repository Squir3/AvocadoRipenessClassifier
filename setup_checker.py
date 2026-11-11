import torch

print(f"Czy CUDA jest dostępna: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Nazwa GPU: {torch.cuda.get_device_name(0)}")
    print(f"Liczba urządzeń: {torch.cuda.device_count()}")
    print(f"Wersja CUDA wg PyTorch: {torch.version.cuda}")
else:
    print("❌ CUDA niewykryta – trening odbywa się na CPU.")
