import torchvision
import os
from collections import Counter
from pathlib import Path

root_dir = '/home/ubuntu/Projects/test/datasets'
os.makedirs(root_dir, exist_ok=True)

print("--- Загрузка датасета FGVC-Aircraft ---")
# Загружаем метаданные (split='all' не всегда доступен, поэтому объединяем trainval и test)
trainval_set = torchvision.datasets.FGVCAircraft(root=root_dir, split='trainval', download=True)
test_set = torchvision.datasets.FGVCAircraft(root=root_dir, split='test', download=True)

# В FGVC-Aircraft метки - это индексы в списке классов (dataset.classes)
all_labels = list(trainval_set._labels) + list(test_set._labels)
classes = trainval_set.classes
num_classes = len(classes)
stats = Counter(all_labels)

print(f"\n[РЕЗУЛЬТАТЫ АНАЛИЗА]")
print(f"Общее количество изображений: {len(all_labels)}")
print(f"Количество уникальных классов (вариантов моделей): {num_classes}")

# Проверка балансировки
counts = stats.values()
is_balanced = len(set(counts)) == 1
print(f"Датасет идеально сбалансирован: {'Да' if is_balanced else 'Нет'}")
print(f"Изображений в каждом классе: {list(counts)[0] if is_balanced else 'Разное количество'}")

print("\n[ПРИМЕРЫ КЛАССОВ И КОЛИЧЕСТВО ФОТО]")
for i in range(min(10, num_classes)):
    print(f" - {classes[i]}: {stats[i]} шт.")

print(f"\nДанные успешно сохранены в: {os.path.abspath(root_dir)}")
