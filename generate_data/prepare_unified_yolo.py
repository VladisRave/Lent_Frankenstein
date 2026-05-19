import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import zipfile
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.config import config


def clean_video_name(x):
    x = str(x).strip().split("/")[0]
    if not x.endswith(".mp4"):
        x += ".mp4"
    return x


def voc_to_yolo(img, xmin, ymin, xmax, ymax):
    h, w = img.shape[:2]

    x_center = ((xmin + xmax) / 2) / w
    y_center = ((ymin + ymax) / 2) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h

    return x_center, y_center, bw, bh


def process_lenta_dataset():
    print("Обработка train датасета хакатона...")

    zip_path = config.DOWNLOADED_DIR / "lenta_dataset.zip"
    extract_dir = config.DOWNLOADED_DIR / "lenta_dataset"

    if not zip_path.exists():
        print("lenta_dataset.zip не найден")
        return

    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    csv_files = list(extract_dir.rglob("*.csv"))
    csv_files = [f for f in csv_files if "sample" not in f.name.lower()]

    dfs = []

    for file in csv_files:
        df = pd.read_csv(file)

        if "wholesale_level_1_coun" in df.columns:
            df.rename(columns={
                "wholesale_level_1_coun": "wholesale_level_1_count"
            }, inplace=True)

        df["filename"] = df["filename"].apply(clean_video_name)

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df["price3_qr"].fillna("нет", inplace=True)
    df["print_datetime"].fillna("нет", inplace=True)
    df = df.dropna()

    for col in ["x_min", "y_min", "x_max", "y_max"]:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    print("Извлечение кадров...")

    all_rows = []

    for video_name in tqdm(df["filename"].unique()):
        sub = df[df["filename"] == video_name].copy()

        sub["frame_timestamp"] = sub["frame_timestamp"].astype(float)

        stem = Path(video_name).stem
        video_path = extract_dir / stem / video_name

        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        target_frames = {
            int(round(t * fps / 1000)): int(t)
            for t in sub["frame_timestamp"].unique()
        }

        current_frame = 0

        save_dir = config.DOWNLOADED_DIR / "all_frames" / stem
        save_dir.mkdir(parents=True, exist_ok=True)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if current_frame in target_frames:
                ts = target_frames[current_frame]
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_path = save_dir / f"{ts}.jpg"
                cv2.imwrite(str(img_path), frame)

            current_frame += 1

        cap.release()

        sub["img_path"] = sub["frame_timestamp"].apply(
            lambda x: str(save_dir / f"{int(x)}.jpg")
        )

        all_rows.append(sub)

    df = pd.concat(all_rows, ignore_index=True)

    print("Конвертация bbox...")

    coords = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img = cv2.imread(row["img_path"])

        if img is None:
            coords.append([None]*4)
            continue

        coords.append(voc_to_yolo(
            img,
            row.x_min,
            row.y_min,
            row.x_max,
            row.y_max
        ))

    df[["x_c", "y_c", "w", "h"]] = pd.DataFrame(coords, index=df.index)

    df = df.dropna()

    print("Split train/val...")

    unique_images = df["img_path"].unique()
    np.random.shuffle(unique_images)

    val_size = int(0.1 * len(unique_images))

    val_images = set(unique_images[:val_size])
    train_images = set(unique_images[val_size:])

    df_train = df[df["img_path"].isin(train_images)]
    df_val = df[df["img_path"].isin(val_images)]

    save_split(df_train, "train")
    save_split(df_val, "val")


def save_split(df, split):
    image_dir = config.UNIFIED_DIR / "images" / split
    label_dir = config.UNIFIED_DIR / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = Path(row["img_path"])
        image_name = img_path.name
        stem = img_path.stem

        dst_img = image_dir / image_name

        if not dst_img.exists():
            shutil.copy(img_path, dst_img)

        label_path = label_dir / f"{stem}.txt"

        with open(label_path, "a") as f:
            f.write(
                f"{config.CLASS_ID} "
                f"{row.x_c} {row.y_c} {row.w} {row.h}\n"
            )


def prepare_unified_dataset():
    print("=" * 50)
    print("Подготовка unified dataset")

    config.create_dirs()

    process_lenta_dataset()

    roboflow_path = config.DOWNLOADED_DIR / "roboflow"

    if roboflow_path.exists():
        print("Обработка Roboflow")

        for split in ["train", "val"]:
            src_images = roboflow_path / split / "images"
            dst_images = config.UNIFIED_DIR / "images" / split

            if src_images.exists():
                for img_file in src_images.glob("*.*"):
                    shutil.copy2(img_file, dst_images / img_file.name)

            src_labels = roboflow_path / split / "labels"
            dst_labels = config.UNIFIED_DIR / "labels" / split

            if src_labels.exists():
                for lbl_file in src_labels.glob("*.txt"):
                    shutil.copy2(lbl_file, dst_labels / lbl_file.name)

    hf_path = config.DOWNLOADED_DIR / "huggingface"

    if hf_path.exists():
        print("Обработка HuggingFace")

        for split in ["train", "val"]:
            split_path = hf_path / split

            if not split_path.exists():
                continue

            for img_file in split_path.glob("*.jpg"):
                shutil.copy2(
                    img_file,
                    config.UNIFIED_DIR / "images" / split / img_file.name
                )

                lbl_file = split_path / f"{img_file.stem}.txt"

                if lbl_file.exists():
                    shutil.copy2(
                        lbl_file,
                        config.UNIFIED_DIR / "labels" / split / lbl_file.name
                    )

    create_data_yaml()


def create_data_yaml():
    root = config.UNIFIED_DIR.resolve()

    yaml_content = f"""
train: {(root / 'images/train').as_posix()}
val: {(root / 'images/val').as_posix()}

names:
  {config.CLASS_ID}: {config.CLASS_NAME}
"""

    yaml_path = config.PROJECT_ROOT / "data.yaml"

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"data.yaml создан: {yaml_path}")


if __name__ == "__main__":
    prepare_unified_dataset()