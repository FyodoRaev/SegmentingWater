# config.py
"""
Файл конфигурации.
Здесь собраны все основные параметры для обучения модели.
"""
import torch
import os

# --- Основные настройки ---
MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
NUM_LABELS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Пути к данным и для сохранения ---
BASE_DATA_DIR = "dataset"
OUTPUT_DIR = "saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Создаем папку для моделей, если ее нет

# --- Параметры датасета и DataLoader'а ---
IMAGE_SIZE = (640, 640)
# Стандартные значения для нормализации ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 8 # Уменьшил с 16, т.к. B5 тяжелая модель, может не влезть в память. Подберите под вашу GPU.

# --- Гиперпараметры обучения ---
NUM_EPOCHS = 200
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 0.75

# --- Настройки для Early Stopping и сохранения ---
PATIENCE = 10         # Сколько эпох ждать улучшения перед остановкой
WARMUP_STEPS = 80     # Шаги для "прогрева" learning rate
SAVE_EVERY_N_EPOCHS = 10 # Как часто сохранять промежуточный чекпоинт