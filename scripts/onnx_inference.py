import cv2
import numpy as np
import onnxruntime as ort
import os

def run_onnx_inference(model_path, image_path, save_dir, conf_threshold=0.25, iou_threshold=0.45):
    # Загрузка сессии
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Подготовка изображения
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Ошибка загрузки: {image_path}")
        return
    
    h_orig, w_orig = original_img.shape[:2]
    
    # Ресайз и нормализация
    input_img = cv2.resize(original_img, (640, 640))
    input_img = input_img.astype(np.float32) / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_img = np.expand_dims(input_img, axis=0)

    # Инференс
    outputs = session.run(None, {session.get_inputs()[0].name: input_img})
    output = outputs[0][0] 

    # Постпроцессинг
    output = output.T 
    boxes, confs = [], []
    
    for row in output:
        prob = row[4] # Вероятность самолета
        if prob > conf_threshold:
            xc, yc, nw, nh = row[:4]
            x1 = int((xc - nw/2) * (w_orig / 640))
            y1 = int((yc - nh/2) * (h_orig / 640))
            x2 = int((xc + nw/2) * (w_orig / 640))
            y2 = int((yc + nh/2) * (h_orig / 640))
            
            boxes.append([x1, y1, x2 - x1, y2 - y1]) 
            confs.append(float(prob))

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, iou_threshold)

    # Отрисовка
    if len(indices) > 0:
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            x, y, w, h = boxes[idx]
            conf = confs[idx]
            
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Airplane: {conf:.2f}"
            cv2.putText(original_img, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Формируем имя файла: берем оригинальное имя и сохраняем в папку onnx_results
    img_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"det_{img_name}")
    
    cv2.imwrite(save_path, original_img)
    print(f"Обработано: {img_name} | Найдено: {len(indices)} | Сохранено в: {save_path}")

if __name__ == "__main__":
    # Настройки путей
    MODEL = '/home/ubuntu/Projects/test/models/yolov8n_airplane.onnx'
    SOURCE_DIR = '/home/ubuntu/Projects/test/results/test/images'
    SAVE_DIR = '/home/ubuntu/Projects/test/results/onnx_results'
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Получаем список первых 5 изображений
    images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in images[:5]:
        full_path = os.path.join(SOURCE_DIR, img_file)
        run_onnx_inference(MODEL, full_path, SAVE_DIR)
