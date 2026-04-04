from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(
    data='/home/ubuntu/Projects/test/datasets/aircraft_family',
    epochs=100,
    imgsz=224,
    batch=64,
    device=0,
    name='yolo_aircraft_family'
)
