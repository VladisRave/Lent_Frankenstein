# Lent_Frankenstein

Проект для автоматического распознавания ценников на видео из торгового зала. Решение включает детекцию ценников, OCR, извлечение текстовых и числовых полей, а также формирование итогового submission-файла для соревнования.

## Описание задачи

Необходимо обработать видеозаписи из магазина и автоматически извлечь информацию с ценников:

название товара,
обычную цену,
цену по карте,
скидочную цену,
штрихкод,
QR-код,
дополнительные поля,
координаты ценника на кадре,
время появления ценника.

Основная цель — преобразовать поток видео в структурированную таблицу для дальнейшего анализа.

## Используемый стек

- Python 3.12
- OpenCV
- YOLOv11
- PaddleOCR
- pyzbar
- pandas
- numpy
- CUDA 12.1+

## Архитектура:

Preprocessing Data -> Detector(YOLOv11) -> Crop Images -> OCR + QR (PaddleOCR + pyzbar) -> Result csv

## Структура проекта

```text
Lent_Frankenstein/
├── generate_data/
│   ├── download_extra.py - скачивание дополнительных данных с Roboflow и HuggingFace
│   ├── generate_dataset.py - генерирование формата датасета для обучения YOLO
│   ├── main.py - основной файл генерации данных
│   └── prepare_unified_yolo.py - объединение всех датасетов и приведение к нужному формату координат (x_center, y_center, width, height)
├── infer/
│   └── pipeline.py - инференс видео: детекция, OCR, и вывод итогового submission.csv
├── utils/
│    └── config.py - конфиг для создания поддиректорий
├── yolo_train/
│   ├── config.py - конфиг с параметрами для обучения YOLO
│   └── train.py - файл для тренировки YOLO
├── requirements.txt
└── README.md
```

## Установка окружения 

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

## Генерация данных

python generate_data/main.py

## Запуск обучения YOLO

python yolo_train/train.py

## Запуск инференса (path_to_video заменить на путь с файлом видео):

python infer/pipeline.py --video_dir path_to_video

## После выполнения создаётся файл:

submission.csv
