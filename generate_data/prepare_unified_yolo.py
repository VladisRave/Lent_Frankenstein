import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from pathlib import Path
from utils.config import config

def prepare_unified_dataset():
    """Объединить данные из Roboflow, HuggingFace и сгенерированные в единый YOLO-датасет"""
    print("=" * 50)
    print("Начинаем объединение данных в единый датасет...")
    
    config.create_dirs()   # создаём все нужные папки
    
    # 1. Roboflow (уже YOLO)
    roboflow_path = config.DOWNLOADED_DIR / "roboflow"
    if roboflow_path.exists():
        print("Обработка данных из Roboflow...")
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
    
    # 2. Hugging Face (только изображения, аннотации пока пропускаем или конвертируем)
    hf_path = config.DOWNLOADED_DIR / "huggingface"
    if hf_path.exists():
        print("Обработка данных из Hugging Face...")
        for split in ["train", "val"]:
            split_path = hf_path / split
            if not split_path.exists():
                continue
            for img_file in split_path.glob("*.jpg"):
                shutil.copy2(img_file, config.UNIFIED_DIR / "images" / split / img_file.name)
                # Если есть аннотации в формате YOLO – скопировать
                lbl_file = split_path / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    shutil.copy2(lbl_file, config.UNIFIED_DIR / "labels" / split / lbl_file.name)
    
    print(f"✅ Датасет объединён в: {config.UNIFIED_DIR}")
    create_data_yaml()


def create_data_yaml():
    root = config.UNIFIED_DIR.resolve()
    yaml_content = f"""
train: { (root / "images/train").as_posix() }
val: { (root / "images/val").as_posix() }

names:
  {config.CLASS_ID}: {config.CLASS_NAME}
"""
    yaml_path = config.PROJECT_ROOT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"✅ Файл конфигурации YOLO создан: {yaml_path}")

if __name__ == "__main__":
    prepare_unified_dataset()