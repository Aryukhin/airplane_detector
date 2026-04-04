import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # Для графиков
import os
import pandas as pd # Для сохранения CSV
import time

# --- Настройки ---
data_dir = '/home/ubuntu/Projects/test/datasets/aircraft_family'
batch_size = 64
num_epochs = 100
num_classes = 70
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Логирование
writer = SummaryWriter('runs/resnet50_experiment')
history = []

# --- Данные ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# --- Модель ---
model = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# --- Цикл обучения ---
print(f"Начинаем обучение ResNet-50 на {device}...")
best_acc = 0.0

for epoch in range(num_epochs):
    epoch_stats = {'epoch': epoch}
    
    for phase in ['train', 'val']:
        if phase == 'train': model.train()
        else: model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        # Запись в TensorBoard
        writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
        writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
        
        epoch_stats[f'{phase}_loss'] = epoch_loss
        epoch_stats[f'{phase}_acc'] = float(epoch_acc)

        print(f'Epoch {epoch}/{num_epochs - 1} | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Сохранение лучшей модели
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "resnet50_best.pth")

    history.append(epoch_stats)

# Сохранение истории в CSV для отчета
df = pd.DataFrame(history)
df.to_csv('resnet_training_history.csv', index=False)

writer.close()
print(f"Обучение завершено. Лучшая точность: {best_acc:.4f}")
