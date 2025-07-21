# utils.py
"""
Вспомогательные функции и классы:
- iou_loss: функция потерь на основе Intersection over Union.
- binary_segmentation_metrics: расчет метрик для бинарной сегментации.
- EarlyStopping: класс для ранней остановки обучения.
"""
import torch
import numpy as np
import os
import datetime


def iou_loss(preds: torch.Tensor, labels: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Расчет IoU (Intersection over Union) Loss.
    """
    # Убираем лишние измерения, если они есть
    preds = preds.squeeze(1) if len(preds.shape) > 3 else preds

    # Расчет пересечения и объединения
    intersection = (preds * labels).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + labels.sum(dim=(1, 2)) - intersection

    # Расчет IoU
    iou = (intersection + smooth) / (union + smooth)

    # IoU Loss = 1 - IoU
    return 1 - iou.mean()


def binary_segmentation_metrics(predictions: np.ndarray, targets: np.ndarray) -> tuple:
    """
    Расчет набора метрик для бинарной сегментации.
    """
    eps = 1e-6

    # Бинаризация предсказаний
    predictions_binary = (predictions > 0.5).astype(np.int32)
    targets_binary = targets.astype(np.int32)

    # TP, FP, FN, TN
    TP = np.sum((predictions_binary == 1) & (targets_binary == 1))
    FP = np.sum((predictions_binary == 1) & (targets_binary == 0))
    FN = np.sum((predictions_binary == 0) & (targets_binary == 1))
    TN = np.sum((predictions_binary == 0) & (targets_binary == 0))

    # Метрики
    accuracy = (TP + TN) / (TP + FP + FN + TN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)

    return accuracy, precision, recall, f1_score, iou, dice, FP, FN, TP, TN


class EarlyStopping:
    """
    Реализует раннюю остановку обучения.
    """

    def __init__(self, patience: int = 3, verbose: bool = False, delta: float = 0, path: str = 'best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model: torch.nn.Module):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.best_loss:.6f} --> {self.best_loss:.6f}). Saving model to {self.path}')
        torch.save(model.state_dict(), self.path)