# model.py
"""
Функция для создания и настройки модели SegFormer и оптимизатора.
"""
import torch
from torch.optim import AdamW
from transformers import SegformerForSemanticSegmentation


def create_model(model_name, num_labels, device, lr, weight_decay):
    """
    Загружает предобученную модель, настраивает слои для fine-tuning'а
    и создает оптимизатор.
    """
    # Загружаем модель
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # Важно для fine-tuning'а на кастомное число классов
    )

    # Перемещаем модель на нужное устройство
    model.to(device)

    # Замораживаем все параметры
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем только "голову" для сегментации
    for param in model.decode_head.parameters():
        param.requires_grad = True

    # Создаем оптимизатор, который будет обновлять только размороженные параметры
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    print("Модель SegFormer и оптимизатор успешно созданы.")
    return model, optimizer