import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from roboflow import Roboflow
from utils.config import config


def download_huggingface_dataset():
    """Скачать датасет с Hugging Face и сохранить в downloaded/huggingface"""
    print("=" * 50)
    print("Начинаем загрузку данных с Hugging Face...")

    try:
        dataset = load_dataset(config.HF_DATASET_NAME)
        save_path = config.DOWNLOADED_DIR / "huggingface"

        for split_name, split_data in dataset.items():
            split_dir = save_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for idx, sample in enumerate(split_data):
                image = sample["image"]
                image_path = split_dir / f"{idx:06d}.jpg"
                image.save(image_path)

                # Если есть аннотации – можно конвертировать в YOLO позже
                # Сейчас просто сохраняем изображения

        print(f"✅ Hugging Face датасет загружен в: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ Ошибка при загрузке с Hugging Face: {e}")
        return None


def download_roboflow_dataset():
    """Скачать датасет с Roboflow в формате YOLO"""
    print("=" * 50)
    print("Начинаем загрузку данных с Roboflow...")

    try:
        rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
        project = rf.workspace(config.ROBOFLOW_WORKSPACE).project(config.ROBOFLOW_PROJECT)
        version = project.version(config.ROBOFLOW_VERSION)

        location = config.DOWNLOADED_DIR / "roboflow"
        version.download("yolov11", location=str(location))
        return location
    except Exception as e:
        print(f"❌ Ошибка при загрузке с Roboflow: {e}")
        return None


if __name__ == "__main__":
    download_huggingface_dataset()
    download_roboflow_dataset()