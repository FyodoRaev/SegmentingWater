 Habaek - https://github.com/HanseonJoo/Habaek + одноименная статья Habaek: https://arxiv.org/pdf/2410.15794

 В общем-то статья сопровождалась кодом, но достаточно нехорошим, так что я его почистил:
 * Разделил на блоки config, utils, dataset, model и train
 * Добавил код для теста и инференса (см ниже)
 * Теперь мы каждые 10 эпох сохраняем "словарь состояния" (state dictionary), это лежит с именами "segformer_epoch_x.pth", 
 * После трейна пытаемся сохранить весь объект модели целиком с помощью pickle (torch.save). Это не только веса, но и вся архитектура, код и т.д. Это менее надежно и может ломаться при изменении кода. Оно хранится под именами "nvidia_best_all.pth"

 В файлике test.py можно позапускать модель c "nvidia_best_all.pth", в inference.py можно попробовать веса разных эпох "segformer_epoch_x.pth"

--------
Некоторые веса лежат здесь https://drive.google.com/drive/folders/1-AZmNp-71CXJFN9HpeUS8P5NtNT-kiyu?usp=sharing
