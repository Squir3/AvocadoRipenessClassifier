from ultralytics import YOLO

# Wczytanie najlepszego wytrenowanego modelu
model = YOLO("runs/detect/yolo_avocado_fixed/weights/best.pt")

# Predykcja na zdjęciach testowych
model.predict(
    source="images/test",   # folder z testowymi obrazami
    save=True,              # zapisuj obrazy z zaznaczonymi detekcjami
    conf=0.25               # minimalny próg pewności
)
