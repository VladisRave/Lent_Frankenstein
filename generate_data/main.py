import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download_extra import download_roboflow_dataset, download_huggingface_dataset
from prepare_unified_yolo import prepare_unified_dataset
from utils.config import config
def run_pipeline():
    print("Запуск конвейера подготовки данных")
    print("=" * 50)
    
    if not download_roboflow_dataset():
        print("Пропускаем из-за ошибки Roboflow")
        return
    if not download_huggingface_dataset():
        print("Пропускаем из-за ошибки Hugging Face")
        return
    prepare_unified_dataset()
    
    print("=" * 50)
    print("Конвейер успешно завершён!")

if __name__ == "__main__":
    run_pipeline()