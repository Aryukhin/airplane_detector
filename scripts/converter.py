from ultralytics import YOLO

model = YOLO('/home/ubuntu/Projects/test/models/yolov8n_aircraft_family_clasify.pt')

# dynamic: True позволяет подавать картинки разного размера (полезно)
# simplify: оптимизирует граф ONNX (удаляет лишние узлы)
path = model.export(format='onnx', imgsz=224)

print(f"Модель успешно сохранена по пути: {path}")
