import os
import shutil
from pathlib import Path

source_dir = Path("Avocado Ripening Dataset")
output_dir = Path("dataset_crop")

# Tworzenie folderów klas 1–5
for i in range(1, 6):
    os.makedirs(output_dir / str(i), exist_ok=True)

# Przetwarzanie plików
for img_path in source_dir.glob("*.jpg"):
    filename = img_path.stem  # np. "Tam_d10_164_b_5"
    try:
        label = int(filename.split("_")[-1])  # ostatni fragment nazwy to klasa
        if 1 <= label <= 5:
            shutil.copy(img_path, output_dir / str(label) / img_path.name)
    except ValueError:
        continue

print("✅ Pliki zostały posortowane do dataset_crop/<klasa>/ na podstawie nazw.")
