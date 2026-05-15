import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from utils.config import config

OUTPUT_DIR = config.PROJECT_ROOT / config.DATA_ROOT / "dataset"   # или оставить как OUTPUT_DIR из .env
SPLIT = "val"
CLASS_ID = config.CLASS_ID

# Создаём папки
(OUTPUT_DIR / "images" / SPLIT).mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels" / SPLIT).mkdir(parents=True, exist_ok=True)

def to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height

# Поиск пар видео/csv
raw_data_root = config.PROJECT_ROOT / config.DATA_ROOT
pairs = []
for folder in raw_data_root.iterdir():
    if not folder.is_dir():
        continue
    mp4 = list(folder.glob("*.mp4"))
    csv = list(folder.glob("*.csv"))
    if mp4 and csv:
        pairs.append((mp4[0], csv[0], folder.name))

print(f"Найдено пар: {len(pairs)}")

sample_counter = 0

for video_path, csv_path, folder_name in pairs:
    print(f"\nОбработка: {folder_name}")
    
    df = pd.read_csv(csv_path)
    # Преобразование координат
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_annotations = defaultdict(list)
    for _, row in df.iterrows():
        try:
            timestamp = float(row["frame_timestamp"])
            frame_num = int(round(timestamp * fps))
            if 0 <= frame_num < total_frames:
                frames_annotations[frame_num].append((
                    int(row["x_min"]), int(row["y_min"]),
                    int(row["x_max"]), int(row["y_max"])
                ))
        except Exception as e:
            print(f"Ошибка в строке: {e}")
            continue
    
    print(f"Уникальных кадров с аннотациями: {len(frames_annotations)}")
    
    for frame_num, bboxes in tqdm(frames_annotations.items(), desc=folder_name):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        
        h, w = frame.shape[:2]
        yolo_lines = []
        for (x1, y1, x2, y2) in bboxes:
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            xc, yc, wn, hn = to_yolo(x1, y1, x2, y2, w, h)
            yolo_lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        
        if not yolo_lines:
            continue
        
        img_name = f"{sample_counter:08d}.jpg"
        img_path = OUTPUT_DIR / "images" / SPLIT / img_name
        cv2.imwrite(str(img_path), frame)
        
        label_name = f"{sample_counter:08d}.txt"
        label_path = OUTPUT_DIR / "labels" / SPLIT / label_name
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
        
        sample_counter += 1
    
    cap.release()

print(f"\nГотово! Сохранено {sample_counter} кадров в {SPLIT}")