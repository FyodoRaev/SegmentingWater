import torch
import os
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from transformers import SegformerForSemanticSegmentation  # Теперь этот импорт ОБЯЗАТЕЛЕН

# --- Шаг 1: Конфигурация ---

# ИЗМЕНЕНО: Укажите путь к файлу с ВЕСАМИ (state_dict), например, к одному из ваших чекпоинтов
MODEL_WEIGHTS_PATH = os.path.join("saved_models", "segformer_epoch_60.pth")
# Или к лучшей модели от EarlyStopping:
# MODEL_WEIGHTS_PATH = os.path.join("saved_models", "best_model_by_val_loss.pth")

TEST_IMAGES_DIR = "test_images"
OUTPUT_DIR = "output_results"  # Папка для сохранения результатов
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def predict_mask_for_image(model, image_path, device):
    """
    Выполняет предсказание маски для одного изображения.
    Эта функция остается без изменений.
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size[::-1]  # (height, width)

    # Трансформации для Segformer
    transform = T.Compose([
        T.Resize((640, 640)),  # Модель обучалась на этом размере
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pixel_values = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

    # Увеличиваем размер логитов до оригинального размера изображения
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=original_size,
        mode='bilinear',
        align_corners=False
    )

    # Получаем маску по самому вероятному классу для каждого пикселя
    pred_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    return image, pred_mask


def main():
    """
    Основная функция для запуска тестирования.
    """
    print(f"Используемое устройство: {DEVICE}")

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"ОШИБКА: Файл с весами модели не найден по пути: {MODEL_WEIGHTS_PATH}")
        return

    # --- НАЧАЛО КЛЮЧЕВЫХ ИЗМЕНЕНИЙ В ЗАГРУЗКЕ МОДЕЛИ ---
    print(f"Загрузка весов модели из файла {MODEL_WEIGHTS_PATH}...")

    # Шаг 1: Определяем базовую модель, как в скрипте обучения.
    # Это создает "скелет" модели с правильной архитектурой.
    model_name = "nvidia/segformer-b5-finetuned-ade-640-640"

    # Создаем экземпляр модели. Важно указать те же параметры, что и при обучении!
    # num_labels=2 (фон, вода), ignore_mismatched_sizes=True (для замены классификатора).
    print(f"1. Создание архитектуры модели '{model_name}'...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    # Шаг 2: Загружаем сохраненный state_dict (веса) из файла.
    print(f"2. Загрузка весов из файла...")
    # map_location=DEVICE гарантирует, что веса будут загружены на правильное устройство (CPU/GPU)
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)

    # Шаг 3: Применяем загруженные веса к нашей созданной модели.
    print("3. Применение весов к архитектуре...")
    model.load_state_dict(state_dict)

    # Шаг 4: Перемещаем модель на нужное устройство и переводим в режим оценки.
    model.to(DEVICE)
    model.eval()  # !!! ОБЯЗАТЕЛЬНО: отключает dropout, batch norm в режиме обучения и т.д.

    print("Модель успешно создана и веса загружены.")
    # --- КОНЕЦ КЛЮЧЕВЫХ ИЗМЕНЕНИЙ ---

    # Поиск всех изображений в тестовой папке
    test_image_paths = glob(os.path.join(TEST_IMAGES_DIR, '*.[jJ][pP][gG]')) + \
                       glob(os.path.join(TEST_IMAGES_DIR, '*.[jJ][pP][eE][gG]')) + \
                       glob(os.path.join(TEST_IMAGES_DIR, '*.[pP][nN][gG]'))

    if not test_image_paths:
        print(f"\nОШИБКА: Изображения для теста не найдены в папке '{TEST_IMAGES_DIR}'.")
        print("Пожалуйста, поместите тестовые изображения в эту папку.")
        return

    print(f"\nНайдено {len(test_image_paths)} изображений для теста. Начинаем обработку...")

    for image_path in test_image_paths:
        try:
            print(f"\nОбработка изображения: {os.path.basename(image_path)}")
            original_image, predicted_mask = predict_mask_for_image(model, image_path, DEVICE)

            # Визуализация и сохранение результата
            plt.figure(figsize=(15, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title("Оригинальное изображение")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(original_image)
            # Накладываем маску полупрозрачным слоем
            plt.imshow(predicted_mask, cmap='viridis', alpha=0.5)
            plt.title("Предсказанная маска (наложение)")
            plt.axis('off')

            plt.tight_layout()

            # Формируем имя выходного файла на основе имени входного
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}_result.png")

            # Сохраняем фигуру в файл
            plt.savefig(save_path)
            print(f"Результат сохранен в: {save_path}")

            # Закрываем фигуру, чтобы не отображать ее в jupyter/spyder и освободить память
            plt.close()

        except Exception as e:
            print(f"Не удалось обработать файл {image_path}. Ошибка: {e}")


if __name__ == '__main__':
    main()