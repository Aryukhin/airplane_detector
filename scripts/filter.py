import os
import shutil
from tqdm import tqdm

src_labels = 'datasets/coco/labels'
src_images = 'datasets/coco/images'
dst_path = 'datasets/airplane_only'

def prepare_airplane_dataset(split):
    label_src = os.path.join(src_labels, split)
    img_src = os.path.join(src_images, split)
    
    label_dst = os.path.join(dst_path, 'labels', split)
    img_dst = os.path.join(dst_path, 'images', split)
    
    os.makedirs(label_dst, exist_ok=True)
    os.makedirs(img_dst, exist_ok=True)

    for label_file in tqdm(os.listdir(label_src), desc=f"Processing {split}"):
        with open(os.path.join(label_src, label_file), 'r') as f:
            lines = f.readlines()
        
        # 4 - это ID самолета в COCO. Меняем его на 0 для нашей новой модели
        airplane_lines = [line.replace('4 ', '0 ', 1) for line in lines if line.startswith('4 ')]
        
        if airplane_lines:
            # Копируем разметку
            with open(os.path.join(label_dst, label_file), 'w') as f:
                f.writelines(airplane_lines)
            
            # Копируем соответствующее изображение
            img_file = label_file.replace('.txt', '.jpg')
            if os.path.exists(os.path.join(img_src, img_file)):
                shutil.copy(os.path.join(img_src, img_file), os.path.join(img_dst, img_file))

# Запускаем для тренировочной и валидационной выборки
prepare_airplane_dataset('train2017')
prepare_airplane_dataset('val2017')

print(f"Готово! Датасет только с самолетами создан в: {dst_path}")
