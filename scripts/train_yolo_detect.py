from ultralytics import YOLO

model = YOLO('../models/yolov8n_custom_v1.yaml')

results = model.train(
    data='airplane.yaml',
    epochs=100,
    imgsz=640,
    batch=64,
    device=0,
    single_cls=True,
    name='yolov8n_airplane_custom_1'
)