from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")  # pretrenowany model YOLOv8 small

    model.train(
        data="data.yaml",
        epochs=30,          #  krótszy trening
        batch=8,            #  bezpiecznie dla RTX 4050
        imgsz=640,
        name="yolo_avocado_fixed",
        device=0,           # użycie GPU
        cache=True,
        rect=True
    )

if __name__ == '__main__':
    main()
