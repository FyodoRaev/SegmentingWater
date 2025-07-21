# dataset.py
"""
Логика работы с данными:
- CustomSegmentationDataset: кастомный датасет для PyTorch.
- get_data_loaders: функция для поиска путей к данным и создания загрузчиков.
"""
import os
from glob import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import BASE_DATA_DIR

class CustomSegmentationDataset(Dataset):
    def __init__(self, image_paths, label_paths, mean, std, target_size=(640, 640)):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.label_paths[idx]).convert("L")

        image = self.transform(image)

        mask = mask.resize(self.target_size, Image.NEAREST)  # Используем NEAREST для масок
        mask = np.array(mask, dtype=np.uint8)
        mask[mask > 0] = 1  # Бинаризация маски
        mask = torch.tensor(mask, dtype=torch.long)

        return {'pixel_values': image, 'labels': mask}


base_dir= BASE_DATA_DIR

def get_data_loaders(batch_size, mean, std, target_size):
    # Определяем пути к датасетам
    WATER_V2_DIR = os.path.join(base_dir, "water-segmentation", "water_v2", "water_v2")
    RIWA_V2_DIR = os.path.join(base_dir, "river-water-segmentation", "riwa_v2")
    LUFI_DIR = os.path.join(base_dir, "lufi-riversnap", "LuFI-RiverSnap.v1")

    # --- Загрузка путей к изображениям ---
    ade20k_image_paths = []
    for ext in ["jpeg", "jpg", "png"]:
        # Используем f-строку и os.path.join для формирования паттерна glob
        pattern = os.path.join(WATER_V2_DIR, "JPEGImages", "ADE20K", "**",
                               f"*.[jJ][pP][gG]" if ext == "jpg" else f"*.[{ext[0].lower()}{ext[0].upper()}][{ext[1].lower()}{ext[1].upper()}][{ext[2].lower()}{ext[2].upper()}]*")
        ade20k_image_paths.extend(glob(pattern, recursive=True))
    ade20k_image_paths = sorted(ade20k_image_paths)

    riwa_train_image_paths = sorted(glob(os.path.join(RIWA_V2_DIR, "images", "**", "*.[jJ][pP][gG]"), recursive=True))
    riwa_val_image_paths = sorted(
        glob(os.path.join(RIWA_V2_DIR, "validation", "images", "**", "*.[jJ][pP][gG]"), recursive=True))

    # Датасет Lufi-RiverSnap
    lufi_train_image_paths = sorted(
        glob(os.path.join(LUFI_DIR, "Train", "Images", "**", "*.[jJ][pP][gG]"), recursive=True))
    lufi_val_image_paths = sorted(glob(os.path.join(LUFI_DIR, "Val", "Images", "**", "*.[jJ][pP][gG]"), recursive=True))
    # --- Загрузка путей к маскам ---
    # Датасет water_v2 (содержит ADE20K)
    ade20k_label_paths = sorted(
        glob(os.path.join(WATER_V2_DIR, "Annotations", "ADE20K", "**", "*.[pP][nN][gG]"), recursive=True))

    # Датасет riwa_v2 ("дополнительный" датасет)
    riwa_train_label_paths = sorted(glob(os.path.join(RIWA_V2_DIR, "masks", "**", "*.[pP][nN][gG]"), recursive=True))
    riwa_val_label_paths = sorted(
        glob(os.path.join(RIWA_V2_DIR, "validation", "masks", "**", "*.[pP][nN][gG]"), recursive=True))

    # Датасет Lufi-RiverSnap
    lufi_train_label_paths = sorted(
        glob(os.path.join(LUFI_DIR, "Train", "Labels", "**", "*.[pP][nN][gG]"), recursive=True))
    # Обратите внимание на 'labels' в нижнем регистре, как в вашей структуре
    lufi_val_label_paths = sorted(glob(os.path.join(LUFI_DIR, "Val", "labels", "**", "*.[pP][nN][gG]"), recursive=True))

    # --- Объединение всех путей в тренировочный и валидационный списки ---

    # Разделяем ADE20K на train и val по наличию "ADE_train" или "ADE_val" в пути
    train_image_paths = [path for path in ade20k_image_paths if "ADE_train" in path]
    val_image_paths = [path for path in ade20k_image_paths if "ADE_val" in path]

    train_label_paths = [path for path in ade20k_label_paths if "ADE_train" in path]
    val_label_paths = [path for path in ade20k_label_paths if "ADE_val" in path]

    # Добавляем пути из других датасетов
    train_image_paths.extend(riwa_train_image_paths)
    val_image_paths.extend(riwa_val_image_paths)
    train_image_paths.extend(lufi_train_image_paths)
    val_image_paths.extend(lufi_val_image_paths)

    train_label_paths.extend(riwa_train_label_paths)
    val_label_paths.extend(riwa_val_label_paths)
    train_label_paths.extend(lufi_train_label_paths)
    val_label_paths.extend(lufi_val_label_paths)
    # --- Проверки ---


    print(f"Найдено для обучения: {len(train_image_paths)} изображений")
    print(f"Найдено для валидации: {len(val_image_paths)} изображений")

    assert len(train_image_paths) > 0, "Не найдены изображения для обучения!"
    assert len(val_image_paths) > 0, "Не найдены изображения для валидации!"
    assert len(train_image_paths) == len(train_label_paths), "Количество изображений и масок для обучения не совпадает!"
    assert len(val_image_paths) == len(val_label_paths), "Количество изображений и масок для валидации не совпадает!"
    # --- Создание датасетов и загрузчиков ---
    train_dataset = CustomSegmentationDataset(train_image_paths, train_label_paths, mean, std, target_size)
    val_dataset = CustomSegmentationDataset(val_image_paths, val_label_paths, mean, std, target_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader