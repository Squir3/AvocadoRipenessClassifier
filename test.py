from ultralytics import YOLO

model = YOLO("runs/detect/yolo_avocado42/weights/best.pt")
model.predict(source="images/test", save=True, conf=0.25)
