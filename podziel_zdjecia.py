import os
import shutil
import random

# === ŚCIEŻKI ===
base_dir = "C:/Users/gadam/OneDrive/Desktop/DeepLearning/Lista4"

# Folder ze zdjęciami
source_images_dir = os.path.join(base_dir, "Avocado Ripening Dataset")
# Folder z adnotacjami .txt (YOLO format)
source_labels_dir = os.path.join(base_dir, "labels_all")

# Docelowe foldery
images_train = os.path.join(base_dir, "images", "train")
images_test = os.path.join(base_dir, "images", "test")
labels_train = os.path.join(base_dir, "labels", "train")
labels_test = os.path.join(base_dir, "labels", "test")

# === TWORZENIE FOLDERÓW ===
os.makedirs(images_train, exist_ok=True)
os.makedirs(images_test, exist_ok=True)
os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_test, exist_ok=True)

# === PLIKI ===
all_files = [f for f in os.listdir(source_images_dir) if f.endswith(".jpg")]
random.seed(42)
random.shuffle(all_files)

# Podział
split = int(0.8 * len(all_files))
train_files = all_files[:split]
test_files = all_files[split:]

# === KOPIOWANIE ===
def copy_files(image_list, img_dst, lbl_dst):
    for img_file in image_list:
        # kopiuj obraz
        shutil.copy(os.path.join(source_images_dir, img_file), os.path.join(img_dst, img_file))

        # nazwa pliku .txt
        txt_file = img_file.replace(".jpg", ".txt")
        txt_path = os.path.join(source_labels_dir, txt_file)

        # jeśli plik z adnotacją istnieje — kopiuj
        if os.path.exists(txt_path):
            shutil.copy(txt_path, os.path.join(lbl_dst, txt_file))

# Kopiuj treningowe i testowe
copy_files(train_files, images_train, labels_train)
copy_files(test_files, images_test, labels_test)

print(f"✔ Skopiowano {len(train_files)} zdjęć + labeli do 'train/'")
print(f"✔ Skopiowano {len(test_files)} zdjęć + labeli do 'test/'")
