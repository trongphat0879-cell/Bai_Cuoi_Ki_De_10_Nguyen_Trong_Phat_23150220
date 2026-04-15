from ultralytics import YOLO

model = YOLO("fire.pt")

print(model.names)