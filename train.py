# train.py
"""
Основной скрипт для запуска процесса обучения модели.
"""
import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

# Импортируем наши модули
import config
from utils import iou_loss, binary_segmentation_metrics, EarlyStopping
from dataset import get_data_loaders
from model import create_model


def train_one_epoch(model, loader, optimizer, scheduler, device, max_grad_norm):
    """Процесс обучения за одну эпоху."""
    model.train()
    epoch_losses = []
    epoch_metrics = []

    for batch in tqdm(loader, desc="Training"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

        # Апскейлинг предсказаний до размера масок
        upsampled_logits = F.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        # Используем sigmoid, т.к. у нас 2 класса (фон/вода), и берем только канал для воды
        preds = torch.sigmoid(upsampled_logits)[:, 1, :, :]

        loss = iou_loss(preds, labels.float())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss.item())

        # Расчет метрик
        preds_np = (preds > 0.5).float().detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        metrics = binary_segmentation_metrics(preds_np, labels_np)
        epoch_metrics.append(metrics)

    return np.mean(epoch_losses), np.mean(epoch_metrics, axis=0)


def evaluate(model, loader, device):
    """Оценка модели на валидационном сете."""
    model.eval()
    val_losses = []
    val_metrics = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            upsampled_logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            preds = torch.sigmoid(upsampled_logits)[:, 1, :, :]

            loss = iou_loss(preds, labels.float())
            val_losses.append(loss.item())

            preds_np = (preds > 0.5).float().cpu().numpy()
            labels_np = labels.cpu().numpy()
            metrics = binary_segmentation_metrics(preds_np, labels_np)
            val_metrics.append(metrics)

    return np.mean(val_losses), np.mean(val_metrics, axis=0)


def main():
    """Основная функция, запускающая весь pipeline."""
    start_time = time.time()

    print(f"Используемое устройство: {config.DEVICE}")

    # 1. Создание модели и оптимизатора
    model, optimizer = create_model(
        model_name=config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        device=config.DEVICE,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 2. Создание загрузчиков данных
    train_loader, val_loader = get_data_loaders(
        batch_size=config.BATCH_SIZE,
        mean=config.MEAN,
        std=config.STD,
        target_size=config.IMAGE_SIZE
    )

    # 3. Настройка планировщика и EarlyStopping
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / config.WARMUP_STEPS) if step < config.WARMUP_STEPS else 1.0
    )
    best_model_path = os.path.join(config.OUTPUT_DIR, 'best_model_by_val_loss.pth')
    early_stopping = EarlyStopping(patience=config.PATIENCE, verbose=True, path=best_model_path)

    # 4. Цикл обучения
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Эпоха {epoch + 1}/{config.NUM_EPOCHS} ---")

        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, config.DEVICE,
                                                    config.MAX_GRAD_NORM)
        val_loss, val_metrics = evaluate(model, val_loader, config.DEVICE)

        # Вывод метрик
        print(
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_metrics[0]:.4f} | Train IoU: {train_metrics[4]:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_metrics[0]:.4f} | Val IoU:   {val_metrics[4]:.4f}")

        # Периодическое сохранение
        if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = os.path.join(config.OUTPUT_DIR, f"segformer_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--- Чекпоинт сохранен в {checkpoint_path} ---")

        # Проверка Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Сработала ранняя остановка.")
            break

    # 5. Сохранение финальной модели
    final_model_path = os.path.join(config.OUTPUT_DIR, 'segformer_final_state_dict.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nФинальные веса модели сохранены в {final_model_path}")
    print(f"Лучшая модель (по val loss) сохранена в {best_model_path}")

    end_time = time.time()
    print(f"Общее время обучения: {(end_time - start_time) / 60:.2f} минут")


if __name__ == '__main__':
    main()