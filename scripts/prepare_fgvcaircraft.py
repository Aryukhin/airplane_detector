import os
import shutil
from pathlib import Path

# Исходные пути (проверь наличие папки data внутри fgvc-aircraft-2013b)
src_root = '/home/ubuntu/Projects/test/datasets/fgvc-aircraft-2013b/data'
output_root = '/home/ubuntu/Projects/test/datasets/aircraft_family'

def prepare_split(split_name):
    # Файл со списком: '0001234 Boeing 737'
    list_file = os.path.join(src_root, f'images_family_{split_name}.txt')
    img_dir = os.path.join(src_root, 'images')
    
    with open(list_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(' ', 1)
        img_id = parts[0]
        family = parts[1].replace(' ', '_').replace('/', '-')
        
        # Создаем папку класса
        dst_class_dir = Path(output_root) / split_name / family
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Копируем файл
        src_img = os.path.join(img_dir, f"{img_id}.jpg")
        dst_img = dst_class_dir / f"{img_id}.jpg"
        
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)

# В датасете есть train, val и test. Для обучения объединим train и val
prepare_split('trainval') # Это будет наша папка train
prepare_split('test')     # Это будет наша папка val (для теста)

# Переименуем для удобства YOLO
os.rename(os.path.join(output_root, 'trainval'), os.path.join(output_root, 'train'))
os.rename(os.path.join(output_root, 'test'), os.path.join(output_root, 'val'))

print(f"Готово! Данные подготовлены в {output_root}")
