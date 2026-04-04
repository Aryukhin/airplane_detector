import shutil
from ultralytics.nn import tasks

# Находим путь к установленному пакету
path = tasks.__file__.replace('tasks.py', 'modules/configs/v8/yolov8.yaml')
# Если путь выше не сработал (зависит от версии), попробуй этот:
# path = "env/lib/python3.12/site-packages/ultralytics/cfg/models/v8/yolov8.yaml"

shutil.copy(path, 'yolov8n_custom.yaml')
print("Файл скопирован!")
