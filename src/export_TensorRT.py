from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='engine', imgsz=800, half=True, dynamic=True, batch=2, device=0)