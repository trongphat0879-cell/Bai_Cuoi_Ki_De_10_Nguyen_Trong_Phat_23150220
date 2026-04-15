from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="D:/Hoc_Phan_HK2_Nam_3/fire_detection/FIREDETECTIONYOLOV8.v3i.yolov8/data.yaml",
    epochs=10,
    imgsz=416
)