# utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные из .env (ищем в корне проекта)
load_dotenv()

class Config:
    # Корень проекта (папка, где лежит этот файл/../)
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

    # ----- Пути -----
    DATA_ROOT = PROJECT_ROOT / os.getenv("DATA_ROOT", "raw_data")
    DOWNLOADED_DIR = DATA_ROOT / "downloaded"
    UNIFIED_DIR = DATA_ROOT / "unified"

    # ----- Roboflow -----
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
    ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT")
    ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", "1"))

    # ----- Hugging Face -----
    HF_DATASET_NAME = os.getenv("HF_DATASET_NAME")

    # ----- YOLO / класс -----
    CLASS_ID = int(os.getenv("CLASS_ID", "0"))
    CLASS_NAME = os.getenv("CLASS_NAME", "price-tag")

    # ----- Разбиение train/val -----
    TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.8"))
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

    @classmethod
    def create_dirs(cls):
        """Создаёт все необходимые поддиректории для unified датасета и загрузок"""
        dirs = [
            cls.DOWNLOADED_DIR / "roboflow",
            cls.DOWNLOADED_DIR / "huggingface",
            cls.UNIFIED_DIR / "images" / "train",
            cls.UNIFIED_DIR / "images" / "val",
            cls.UNIFIED_DIR / "labels" / "train",
            cls.UNIFIED_DIR / "labels" / "val",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

config = Config()