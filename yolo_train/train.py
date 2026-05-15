from ultralytics import YOLO
import torch

import config

# =========================
# GPU INFO
# =========================

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =========================
# LOAD MODEL
# =========================

model = YOLO(config.MODEL_NAME)

# =========================
# TRAIN
# =========================

model.train(
    data="data.yaml",

    epochs=config.EPOCHS,

    imgsz=config.IMAGE_SIZE,

    batch=config.BATCH_SIZE,

    device=0,

    workers=4,

    project="runs",

    name="price_detector",

    pretrained=True,

    optimizer="AdamW",

    lr0=0.005,

    patience=7,

    save=True,

    save_period=20,

    cache=False,

    amp=True,

    cos_lr=True,

    close_mosaic=8,

    degrees=3,
  
    translate=0.1,
  
    scale=0.5,

    shear=0.0,

    perspective=0.0005,

    flipud=0.0,

    fliplr=0.5,

    mosaic=1.0,

    mixup=0.1,

    copy_paste=0.0
)

# VALIDATION

metrics = model.val()

print(metrics)