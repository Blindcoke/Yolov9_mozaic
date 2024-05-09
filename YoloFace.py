from ultralytics import YOLO
model = YOLO("yolov8n-face.pt")
results = model.predict(source="test.mp4", show= True)
