import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from tqdm import tqdm
from glob import glob
from PIL import Image
import cv2
import os
from torchvision import transforms
import time

model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모든 파라미터를 고정(freeze)
for param in model.parameters():
    param.requires_grad = False

# Segmentation Head 부분만 학습 가능하게 설정
for param in model.decode_head.parameters():
    param.requires_grad = True

# 옵티마이저 설정 (고정된 파라미터는 제외하고 활성화된 파라미터만 업데이트)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)

def iou_loss(preds, labels, smooth=1e-6):
    # 예측값과 실제 라벨의 교집합과 합집합 계산
    intersection = (preds * labels).sum(dim=(1, 2))  # 배치 차원 제외한 후 계산
    union = preds.sum(dim=(1, 2)) + labels.sum(dim=(1, 2)) - intersection

    # IoU 계산 (교집합 / 합집합)
    iou = (intersection + smooth) / (union + smooth)

    # IoU 손실은 1 - IoU
    return 1 - iou.mean()

# ImageNet Mean and Std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

print(f"Using ImageNet mean: {mean}")
print(f"Using ImageNet std: {std}")

# Custom 데이터셋 클래스 (계산한 mean과 std를 사용하여 정규화)
class CustomSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, label_paths, mean, std, target_size=(640, 640)):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.target_size = target_size

        # 이미지 전처리: 정규화 포함
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),  # 텐서 변환
            transforms.Normalize(mean=mean, std=std)  # 계산한 mean, std로 정규화
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지와 마스크 로드
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.label_paths[idx]).convert("L")

        # 이미지 전처리
        image = self.transform(image)

        # 마스크를 0 또는 1로 변환
        mask = mask.resize(self.target_size)
        mask = np.array(mask)
        mask[mask > 0] = 1
        mask = torch.tensor(mask, dtype=torch.long)

        return {'pixel_values': image, 'labels': mask}

import os
from glob import glob
from torch.utils.data import DataLoader

# --- НАЧАЛО ИЗМЕНЕНИЙ ---

# 1. Определяем базовую директорию для датасетов.
#    Предполагается, что скрипт лежит рядом с папкой 'dataset'.
# Добавьте это в начало вашего файла, после определения путей к данным

# --- Конфигурация путей для сохранения модели ---
OUTPUT_DIR = "saved_models"
# Создаем директорию, если она не существует. exist_ok=True предотвращает ошибку, если папка уже есть.
os.makedirs(OUTPUT_DIR, exist_ok=True)
BASE_DATA_DIR = "dataset"

# 2. Определяем пути к каждому конкретному датасету для чистоты кода.
#    Используем os.path.join для кросс-платформенной совместимости.
WATER_V2_DIR = os.path.join(BASE_DATA_DIR, "water-segmentation", "water_v2", "water_v2")
RIWA_V2_DIR = os.path.join(BASE_DATA_DIR, "river-water-segmentation", "riwa_v2")
LUFI_DIR = os.path.join(BASE_DATA_DIR, "lufi-riversnap", "LuFI-RiverSnap.v1")

# --- Загрузка путей к изображениям ---

# Датасет water_v2 (содержит ADE20K)
# Собираем пути для всех форматов изображений из папки ADE20K
ade20k_image_paths = []
for ext in ["jpeg", "jpg", "png"]:
    # Используем f-строку и os.path.join для формирования паттерна glob
    pattern = os.path.join(WATER_V2_DIR, "JPEGImages", "ADE20K", "**", f"*.[jJ][pP][gG]" if ext == "jpg" else f"*.[{ext[0].lower()}{ext[0].upper()}][{ext[1].lower()}{ext[1].upper()}][{ext[2].lower()}{ext[2].upper()}]*")
    ade20k_image_paths.extend(glob(pattern, recursive=True))
ade20k_image_paths = sorted(ade20k_image_paths)


# Датасет riwa_v2 ("дополнительный" датасет)
# В вашей структуре нет папки 'train', поэтому берем из корневой папки 'images'
riwa_train_image_paths = sorted(glob(os.path.join(RIWA_V2_DIR, "images", "**", "*.[jJ][pP][gG]"), recursive=True))
riwa_val_image_paths = sorted(glob(os.path.join(RIWA_V2_DIR, "validation", "images", "**", "*.[jJ][pP][gG]"), recursive=True))

# Датасет Lufi-RiverSnap
lufi_train_image_paths = sorted(glob(os.path.join(LUFI_DIR, "Train", "Images", "**", "*.[jJ][pP][gG]"), recursive=True))
lufi_val_image_paths = sorted(glob(os.path.join(LUFI_DIR, "Val", "Images", "**", "*.[jJ][pP][gG]"), recursive=True))

# --- Загрузка путей к маскам (лейблам) ---

# Датасет water_v2 (содержит ADE20K)
ade20k_label_paths = sorted(glob(os.path.join(WATER_V2_DIR, "Annotations", "ADE20K", "**", "*.[pP][nN][gG]"), recursive=True))

# Датасет riwa_v2 ("дополнительный" датасет)
riwa_train_label_paths = sorted(glob(os.path.join(RIWA_V2_DIR, "masks", "**", "*.[pP][nN][gG]"), recursive=True))
riwa_val_label_paths = sorted(glob(os.path.join(RIWA_V2_DIR, "validation", "masks", "**", "*.[pP][nN][gG]"), recursive=True))

# Датасет Lufi-RiverSnap
lufi_train_label_paths = sorted(glob(os.path.join(LUFI_DIR, "Train", "Labels", "**", "*.[pP][nN][gG]"), recursive=True))
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


# --- КОНЕЦ ИЗМЕНЕНИЙ ---

# Остальная часть кода остается без изменений

# 데이터셋 확인
print(f"Train images: {len(train_image_paths)}")
print(f"Val images: {len(val_image_paths)}")
print(f"Train labels: {len(train_label_paths)}")
print(f"Val labels: {len(val_label_paths)}")

# ImageNet Mean and Std (혹은 다른 mean/std 값을 사용할 수 있습니다)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Предполагается, что класс CustomSegmentationDataset уже где-то определен
# from your_dataset_file import CustomSegmentationDataset

# 트레인 데이터셋 로드
# train_dataset = CustomSegmentationDataset(train_image_paths, train_label_paths, mean=mean, std=std)

# Validation 데이터셋 로드
# val_dataset = CustomSegmentationDataset(val_image_paths, val_label_paths, mean=mean, std=std)

# DataLoader 설정 (배치 크기 16, 셔플 있음/없음)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 데이터가 잘 로드되었는지 확인
assert len(train_image_paths) > 0, "Train image paths are empty!"
assert len(val_image_paths) > 0, "Val image paths are empty!"
assert len(train_image_paths) == len(train_label_paths), "Train image and label counts do not match!"
assert len(val_image_paths) == len(val_label_paths), "Val image and label counts do not match!"

print("\nПути к данным успешно сконфигурированы и проверены!")


# 데이터셋 확인
print(f"Train images: {len(train_image_paths)}")
print(f"Val images: {len(val_image_paths)}")
print(f"Train labels: {len(train_label_paths)}")
print(f"Val labels: {len(val_label_paths)}")

# ImageNet Mean and Std (혹은 다른 mean/std 값을 사용할 수 있습니다)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 트레인 데이터셋 로드
train_dataset = CustomSegmentationDataset(train_image_paths, train_label_paths, mean=mean, std=std)

# Validation 데이터셋 로드
val_dataset = CustomSegmentationDataset(val_image_paths, val_label_paths, mean=mean, std=std)

# DataLoader 설정 (배치 크기 16, 셔플 있음/없음)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 데이터가 잘 로드되었는지 확인
assert len(train_image_paths) > 0, "Train image paths are empty!"
assert len(val_image_paths) > 0, "Val image paths are empty!"
assert len(train_image_paths) == len(train_label_paths), "Train image and label counts do not match!"
assert len(val_image_paths) == len(val_label_paths), "Val image and label counts do not match!"

def binary_segmentation_metrics(predictions, targets):
    """
    예측된 마스크와 실제 마스크를 비교하여 다양한 성능 지표를 계산합니다.
    predictions: torch 텐서 또는 numpy 배열 (예측 마스크)
    targets: torch 텐서 또는 numpy 배열 (실제 마스크)
    """
    # torch 텐서가 들어온 경우 numpy 배열로 변환
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.squeeze().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # 이진화 처리 (0 또는 1)
    predictions_binary = (predictions > 0.5).astype(int)
    targets_binary = targets.astype(int)

    # True Positives, False Positives, False Negatives, True Negatives 계산
    TP = np.sum((predictions_binary == 1) & (targets_binary == 1))
    FP = np.sum((predictions_binary == 1) & (targets_binary == 0))
    FN = np.sum((predictions_binary == 0) & (targets_binary == 1))
    TN = np.sum((predictions_binary == 0) & (targets_binary == 0))

    # 성능 지표 계산
    eps = 1e-6  # 0으로 나누는 상황을 방지하기 위한 epsilon
    accuracy = (TP + TN) / (TP + FP + FN + TN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)

    # 결과 반환 (10개)
    return accuracy, precision, recall, f1_score, iou, dice, FP, FN, TP, TN

# train_on_batch 함수 수정
def train_on_batch(batch):
    batch_losses = []
    batch_metrics = []

    # 배치에서 데이터 추출
    pixel_values = batch['pixel_values'].to(device)  # 이미지를 GPU로 이동
    labels = batch['labels'].to(device)  # 라벨을 GPU로 이동

    # 모델 출력 (SegFormer는 입력에 대해 직접 마스크를 예측함)
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    preds = torch.sigmoid(logits)

    # 예측값을 실제 마스크 크기와 맞추기
    preds = torch.nn.functional.interpolate(preds, size=labels.shape[-2:], mode="bilinear", align_corners=False)

    # 이진 마스크로 변환 (두 번째 차원은 클래스 차원)
    preds = preds[:, 1, :, :]  # 두 번째 클래스 (foreground) 사용
    preds = preds.squeeze(1)  # 불필요한 차원 제거하여 (batch_size, height, width) 형태로 만듦

    # IoU 손실 함수 적용
    loss = iou_loss(preds, labels)

    # 옵티마이저 초기화 및 손실 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_losses.append(loss.item())

    # 성능 지표 계산 (binary_segmentation_metrics 함수 활용)
    preds_np = (preds > 0.5).float().detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    metrics = binary_segmentation_metrics(preds_np, labels_np)
    batch_metrics.append(metrics)

    return batch_losses, batch_metrics

import torch
import numpy as np
import os
import datetime
from tqdm import tqdm

# EarlyStopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, path='best_model' + str(datetime.datetime.now())+ '.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        if self.verbose:
            print(f'Validation loss decreased, saving model...')
        torch.save(model.state_dict(), self.path)

# 검증 데이터셋에서 성능 평가 함수
def evaluate_on_val_set():
    model.eval()  # 평가 모드 설정
    val_losses = []
    val_metrics = []

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            preds = torch.sigmoid(logits)

            preds = torch.nn.functional.interpolate(preds, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = preds[:, 1, :, :]  # 두 번째 클래스 (foreground)
            preds = preds.squeeze(1)

            # IoU 손실 함수 적용
            loss = iou_loss(preds, labels)
            val_losses.append(loss.item())

            preds_np = (preds > 0.4).float().cpu().numpy()
            labels_np = labels.cpu().numpy()
            metrics = binary_segmentation_metrics(preds_np, labels_np)
            val_metrics.append(metrics)

    avg_val_loss = np.mean(val_losses)
    avg_val_metrics = np.mean(val_metrics, axis=0)

    return avg_val_loss, avg_val_metrics

import time


def train_model(num_epochs, patience=3, warmup_steps=100, max_grad_norm=0.8, save_every_n_epochs=10):
    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    # Мы будем использовать OUTPUT_DIR, который был определен ранее в коде.
    # Убедимся, что он существует.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # EarlyStopping по-прежнему полезен для нахождения лучшей модели по val_loss
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model_by_val_loss.pth')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=best_model_path)
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 1.0
    )

    for epoch in range(num_epochs):
        model.train()
        train_losses, train_metrics = [], []

        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
            losses, metrics = train_on_batch(batch)
            train_losses.extend(losses)
            train_metrics.extend(metrics)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scheduler.step()

        avg_train_loss = np.mean(train_losses)
        avg_train_metrics = np.mean(train_metrics, axis=0)
        train_accuracy, train_precision, train_recall, train_f1, train_iou, train_dice, train_fp, train_fn, train_tp, train_tn = avg_train_metrics

        avg_val_loss, avg_val_metrics = evaluate_on_val_set()
        val_accuracy, val_precision, val_recall, val_f1, val_iou, val_dice, val_fp, val_fn, val_tp, val_tn = avg_val_metrics

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        print(
            f"Train Loss (IoU): {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss (IoU):   {avg_val_loss:.4f} | Val Accuracy:   {val_accuracy:.4f} | Val IoU:   {val_iou:.4f}")

        # --- НАЧАЛО ИЗМЕНЕНИЙ: Периодическое сохранение весов ---
        # Проверяем, нужно ли сохранять модель в эту эпоху.
        # (epoch + 1) потому что эпохи для пользователя считаются с 1, а range() с 0.
        if (epoch + 1) % save_every_n_epochs == 0:
            # Формируем путь для сохранения чекпоинта
            checkpoint_path = os.path.join(OUTPUT_DIR, f"segformer_epoch_{epoch + 1}.pth")
            # Сохраняем только веса (state_dict) - это лучшая практика
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--- Checkpoint saved at epoch {epoch + 1} to {checkpoint_path} ---")
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

# 전체 실행 시간 측정
# 전체 실행 시간 측정
start_time = time.time()  # 시작 시간 기록

# Вызываем обучение с новым параметром
train_model(
    num_epochs=200,
    patience=10,  # Увеличил терпение, чтобы early stopping не сработал слишком рано
    warmup_steps=80,
    max_grad_norm=0.75,
    save_every_n_epochs=10  # <-- Вот он, наш новый параметр!
)

end_time = time.time()  # 종료 시간 기록
print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")

# --- Сохранение финальной модели ---
# После окончания обучения (по количеству эпох или из-за EarlyStopping)
# можно сохранить последнюю версию модели.
final_model_path = os.path.join(OUTPUT_DIR, 'segformer_final_state_dict.pth')
torch.save(model.state_dict(), final_model_path)
print(f"\nFinal model's state_dict saved to {final_model_path}")
print(f"Best model based on validation loss saved to {os.path.join(OUTPUT_DIR, 'best_model_by_val_loss.pth')}")


def get_filename_without_extension(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def check_matching_filenames(image_paths, label_paths):
    image_names = [get_filename_without_extension(path) for path in image_paths]
    label_names = [get_filename_without_extension(path) for path in label_paths]

    # 이름이 일치하는 파일 쌍
    matching_pairs = set(image_names) & set(label_names)

    # 일치하지 않는 이미지와 라벨
    non_matching_images = set(image_names) - set(label_names)
    non_matching_labels = set(label_names) - set(image_names)

    # 결과 출력
    if len(non_matching_images) > 0:
        print(f"일치하지 않는 이미지 파일: {non_matching_images}")
    if len(non_matching_labels) > 0:
        print(f"일치하지 않는 라벨 파일: {non_matching_labels}")

    if len(matching_pairs) == len(image_paths) == len(label_paths):
        print("모든 이미지와 라벨 파일이 이름으로 쌍을 이루고 있습니다.")
    else:
        print(f"매칭되지 않은 파일이 있습니다. 총 매칭된 쌍: {len(matching_pairs)}")


import os
import numpy as np
import torch
from tqdm import tqdm
import time


# # 테스트셋 평가 함수
# def evaluate_on_test_set():
#     model.eval()  # 평가 모드 설정
#     test_losses = []
#     test_metrics = []
#     total_images = 0  # 총 이미지 수
#     total_time = 0  # 총 시간
#
#     # tqdm을 사용한 진행 표시
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
#             start_time = time.time()  # 배치 시작 시간
#
#             pixel_values = batch['pixel_values'].to(device)
#             labels = batch['labels'].to(device)
#
#             outputs = model(pixel_values=pixel_values)
#             logits = outputs.logits
#             preds = torch.sigmoid(logits)
#
#             # 이미지 크기에 맞게 예측값 보간 (interpolation)
#             preds = torch.nn.functional.interpolate(preds, size=labels.shape[-2:], mode="bilinear", align_corners=False)
#             preds = preds[:, 1, :, :]  # 두 번째 클래스 (foreground)
#             preds = preds.squeeze(1)
#
#             # IoU 손실 함수 적용
#             loss = iou_loss(preds, labels)
#             test_losses.append(loss.item())  # val_losses -> test_losses로 수정
#
#             # 예측값을 바이너리로 변환
#             preds_np = (preds > 0.5).float().cpu().numpy()
#             labels_np = labels.cpu().numpy()
#
#             # 성능 지표 계산 (binary_segmentation_metrics 함수 활용)
#             metrics = binary_segmentation_metrics(preds_np, labels_np)
#             test_metrics.append(metrics)
#
#             # 처리 시간 계산
#             end_time = time.time()
#             batch_time = end_time - start_time
#             total_time += batch_time
#             total_images += pixel_values.shape[0]  # 배치에 있는 이미지 수 더하기
#
#             # GPU 캐시 삭제
#             torch.cuda.empty_cache()
#
#     # 평균 손실 및 평균 성능 지표 계산
#     avg_test_loss = np.mean(test_losses)
#     avg_test_metrics = np.mean(test_metrics, axis=0)
#
#     # 성능 지표 언패킹
#     accuracy, precision, recall, f1_score, iou, dice, fp, fn, tp, tn = avg_test_metrics
#
#     # 결과 출력
#     print(f"Test Loss: {avg_test_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1_score}, IoU: {iou}, Dice: {dice}")
#     print(f"FP: {fp}, FN: {fn}, TP: {tp}, TN: {tn}")
#
#     # 배치당 처리 시간 및 이미지당 처리 시간 출력
#     avg_batch_time = total_time / len(test_loader)
#     avg_image_time = total_time / total_images
#     print(f"Avg time per batch: {avg_batch_time:.4f} seconds")
#     print(f"Avg time per image: {avg_image_time:.4f} seconds")

# 테스트 데이터셋 평가 실행
# evaluate_on_test_set()

# --- Сохранение модели ---
# 1. Сохранение всей модели целиком
full_model_path = os.path.join(OUTPUT_DIR, 'nvidia_best_all.pth')
torch.save(model, full_model_path)
print(f"Model saved to {full_model_path}")

# 2. Сохранение только весов модели (рекомендуемый способ)
state_dict_path = os.path.join(OUTPUT_DIR, 'nvidia_best_all_state_dict.pth')
torch.save(model.state_dict(), state_dict_path)
print(f"Model's state_dict saved to {state_dict_path}")

# import matplotlib.pyplot as plt
#
# def compare_predictions_with_ground_truth():
#      model.eval()  # 평가 모드로 설정
#      test_image_paths = sorted(glob("/content/drive/MyDrive/Lufi/Test/Images/*.[jJ][pP][gG]"))
#      test_label_paths = sorted(glob("/content/drive/MyDrive/Lufi/Test/Labels/*.[pP][nN][gG]"))
#
#      with torch.no_grad():
#          for idx, batch in enumerate(tqdm(test_loader, desc="Evaluating on Test Set", unit="batch")):
#              pixel_values = batch['pixel_values'].to(device)
#              labels = batch['labels'].to(device)
#
#              outputs = model(pixel_values=pixel_values)
#              logits = outputs.logits
#              preds = torch.sigmoid(logits)
#
#              # 예측된 마스크 크기를 실제 마스크 크기로 맞춤
#              preds = torch.nn.functional.interpolate(preds, size=labels.shape[-2:], mode="bilinear", align_corners=False)
#              preds = preds[:, 1, :, :]  # 두 번째 클래스 사용 (foreground)
#
#              # 시각화: 각 배치의 예측 마스크와 실제 마스크 비교
#              for i in range(len(preds)):
#                  actual_idx = idx * test_loader.batch_size + i
#
#                  # **범위를 체크하여 오류 방지**
#                  if actual_idx >= len(test_image_paths):
#                      continue  # 범위를 넘으면 패스
#
#                  # Ground Truth 마스크 로드
#                  gt_mask_path = test_label_paths[actual_idx]
#                  gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
#
#                  # 예측 마스크를 numpy 형식으로 변환 및 이진화
#                  pred_mask = (preds[i] > 0.5).cpu().numpy().astype(np.uint8)
#
#                  # 시각화
#                  plt.figure(figsize=(12, 4))
#
#                  # 원본 이미지
#                  image = Image.open(test_image_paths[actual_idx])
#                  plt.subplot(1, 3, 1)
#                  plt.imshow(image)
#                  plt.title("Input Image")
#
#                  # 실제 Ground Truth 마스크
#                  plt.subplot(1, 3, 2)
#                  plt.imshow(gt_mask, cmap='gray')
#                  plt.title("Ground Truth Mask")
#
#                  # 예측된 마스크
#                  plt.subplot(1, 3, 3)
#                  plt.imshow(pred_mask, cmap='gray')
#                  plt.title("Predicted Mask")
#
#                  plt.show()

# # 테스트셋에서 예측된 마스크와 실제 마스크 비교
# compare_predictions_with_ground_truth()
