import cv2
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path

def run_cascade_inference(det_session, cls_session, img_path, save_dir, cls_names):
    # Загрузка и подготовка
    original_img = cv2.imread(str(img_path))
    if original_img is None: return
    h_orig, w_orig = original_img.shape[:2]
    
    # Подготовка для Детектора (640x640)
    blob_det = cv2.resize(original_img, (640, 640)).astype(np.float32) / 255.0
    blob_det = blob_det.transpose(2, 0, 1)[np.newaxis, ...]

    # Детекция
    outputs = det_session.run(None, {det_session.get_inputs()[0].name: blob_det})
    output = outputs[0][0].T # (8400, 5)

    boxes, confs = [], []
    for row in output:
        prob = row[4]
        if prob > 0.3: # Порог уверенности детекции
            xc, yc, nw, nh = row[:4]
            x1 = int((xc - nw/2) * (w_orig / 640))
            y1 = int((yc - nh/2) * (h_orig / 640))
            x2 = int((xc + nw/2) * (w_orig / 640))
            y2 = int((yc + nh/2) * (h_orig / 640))
            
            # Ограничение рамки границами кадра
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)
            
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confs.append(float(prob))

    indices = cv2.dnn.NMSBoxes(boxes, confs, 0.3, 0.45)

    # Классификация кропов
    if len(indices) > 0:
        for i in indices:
            idx = i if isinstance(i, (list, np.ndarray)) else i
            x, y, w, h = boxes[idx]
            
            # Вырезаем объект (Crop)
            crop = original_img[y:y+h, x:x+w]
            if crop.size == 0: continue
            
            # Подготовка для Классификатора (224x224)
            blob_cls = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
            blob_cls = blob_cls.transpose(2, 0, 1)[np.newaxis, ...]
            
            # Инференс классификации
            cls_out = cls_session.run(None, {cls_session.get_inputs()[0].name: blob_cls})
            cls_id = np.argmax(cls_out[0])
            family_label = cls_names[cls_id]
            conf_det = confs[idx]

            # Отрисовка результата
            label = f"{family_label} ({conf_det:.2f})"
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(original_img, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Сохранение
    save_path = os.path.join(save_dir, f"final_{os.path.basename(img_path)}")
    cv2.imwrite(save_path, original_img)
    print(f"Processed: {os.path.basename(img_path)} -> Saved to {save_path}")

if __name__ == "__main__":
    # Настройка путей к ONNX моделям
    DET_MODEL = '/home/ubuntu/Projects/test/models/yolov8n_airplane.onnx'
    CLS_MODEL = '/home/ubuntu/Projects/test/models/yolov8n_aircraft_family_clasify.onnx' 
    
    # Пути к данным
    SOURCE_DIR = '/home/ubuntu/Projects/test/results/test_pipeline'
    SAVE_DIR = '/home/ubuntu/Projects/test/results/onnx_pipeline_results'
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_dir = '/home/ubuntu/Projects/test/datasets/aircraft_family/train'
    cls_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    # Инициализация сессий
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    det_sess = ort.InferenceSession(DET_MODEL, providers=providers)
    cls_sess = ort.InferenceSession(CLS_MODEL, providers=providers)

    images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in images[:5]:
        run_cascade_inference(det_sess, cls_sess, os.path.join(SOURCE_DIR, img_file), SAVE_DIR, cls_names)
