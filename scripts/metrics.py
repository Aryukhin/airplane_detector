from ultralytics import YOLO
import pandas as pd

# Пути к твоим лучшим весам
models_paths = {
    "Baseline": "/home/ubuntu/Projects/test/models/yolov8n_airplane.pt",
    "Custom_V1": "/home/ubuntu/Projects/test/models/yolov8_airplane_custom_1.pt",
    "Custom_V2": "/home/ubuntu/Projects/test/models/yolov8_airplane_custom_2.pt"
}

results = []

for name, path in models_paths.items():
    print(f"--- Тестирование {name} ---")
    model = YOLO(path)
    
    # Запускаем валидацию на GPU
    metrics = model.val(data='/home/ubuntu/Projects/test/configs/airplane.yaml', split='val', device=0)
    
    results.append({
        "Model": name,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Inference_ms": metrics.speed['inference'], # Чистое время нейросети
        "Preprocess_ms": metrics.speed['preprocess'],
        "Postprocess_ms": metrics.speed['postprocess'],
        "Params_M": model.info()[0] / 1e6, # Миллионы параметров
        "GFLOPs": model.info()[3]
    })

# Сохраняем в таблицу для отчета
df = pd.DataFrame(results)
df.to_csv("model_comparison.csv", index=False)
print(df)
