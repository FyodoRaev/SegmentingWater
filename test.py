import torch
import os
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# Не нужно импортировать SegformerForSemanticSegmentation, так как мы загрузим его из файла,
# но лучше оставить для ясности и на случай, если захотите переключиться обратно.
from transformers import SegformerForSemanticSegmentation

# --- Шаг 1: Конфигурация ---

# Указываем путь к файлу, который содержит ВСЮ МОДЕЛЬ
MODEL_WEIGHTS_PATH = os.path.join("saved_models", "nvidia_best_all.pth")
TEST_IMAGES_DIR = "test_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(TEST_IMAGES_DIR, exist_ok=True)


def predict_mask_for_image(model, image_path, device):
    """
    Выполняет предсказание маски для одного изображения.
    Эта функция остается без изменений.
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size[::-1]

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pixel_values = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=original_size,
        mode='bilinear',
        align_corners=False
    )

    pred_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    return image, pred_mask


def main():
    """
    Основная функция для запуска тестирования.
    """
    print(f"Используемое устройство: {DEVICE}")

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"ОШИБКА: Файл с моделью не найден по пути: {MODEL_WEIGHTS_PATH}")
        return

    # --- ИЗМЕНЕНИЕ В ЛОГИКЕ ЗАГРУЗКИ МОДЕЛИ ---
    print(f"Загрузка ЦЕЛОЙ МОДЕЛИ из файла {MODEL_WEIGHTS_PATH}...")

    # Мы больше не создаем модель через .from_pretrained() и не используем .load_state_dict().
    # Вместо этого мы просто загружаем весь объект модели из файла.
    # PyTorch сам разберется, как его восстановить.
    model = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)

    # Важные шаги после загрузки, которые нужно сохранить:
    model.to(DEVICE)  # Убедимся, что модель на нужном устройстве
    model.eval()  # !!! Переводим модель в режим оценки

    print("Модель успешно загружена.")
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    test_image_paths = glob(os.path.join(TEST_IMAGES_DIR, '*.[jJ][pP][gG]')) + \
                       glob(os.path.join(TEST_IMAGES_DIR, '*.[jJ][pP][eE][gG]')) + \
                       glob(os.path.join(TEST_IMAGES_DIR, '*.[pP][nN][gG]'))

    if not test_image_paths:
        print(f"\nОШИБКА: Изображения для теста не найдены в папке '{TEST_IMAGES_DIR}'.")
        return

    print(f"Найдено {len(test_image_paths)} изображений для теста. Начинаем обработку...")

    for image_path in test_image_paths:
        print(f"\nОбработка изображения: {os.path.basename(image_path)}")
        original_image, predicted_mask = predict_mask_for_image(model, image_path, DEVICE)

        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Оригинальное изображение")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask, cmap='viridis')
        plt.title("Предсказанная маска воды")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()