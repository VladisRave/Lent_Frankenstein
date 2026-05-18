import cv2
import pandas as pd
import numpy as np
import re
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode
from tqdm import tqdm

VIDEO_DIR = Path("lenta_dataset/Unlabeled")
MODEL_PATH = "best.pt"
OUTPUT_CSV = "submission.csv"

SAMPLE_EVERY_SEC = 2.0

reader = PaddleOCR(
    lang='ru',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

model = YOLO(MODEL_PATH)

PRICE_RE = re.compile(r"\d+[,.]\d{2}")
BARCODE_RE = re.compile(r"\d{8,14}")

COLUMNS = [
    "filename","product_name","price_default","price_card","price_discount",
    "barcode","discount_amount","id_sku","print_datetime","code",
    "additional_info","color","special_symbols",
    "frame_timestamp","x_min","y_min","x_max","y_max",
    "qr_code_barcode","price1_qr","price2_qr","price3_qr","price4_qr",
    "wholesale_level_1_count","wholesale_level_1_price",
    "wholesale_level_2_count","wholesale_level_2_price",
    "action_price_qr","action_code_qr"
]

QR_KEYS = {
    "barcode":"qr_code_barcode",
    "b":"qr_code_barcode",
    "price1":"price1_qr",
    "p1":"price1_qr",
    "price2":"price2_qr",
    "p2":"price2_qr",
    "price3":"price3_qr",
    "p3":"price3_qr",
    "price4":"price4_qr",
    "p4":"price4_qr",
    "wholesaleLevel1Count":"wholesale_level_1_count",
    "wL1C":"wholesale_level_1_count",
    "wholesaleLevel1Price":"wholesale_level_1_price",
    "wL1P":"wholesale_level_1_price",
    "wholesaleLevel2Count":"wholesale_level_2_count",
    "wL2C":"wholesale_level_2_count",
    "wholesaleLevel2Price":"wholesale_level_2_price",
    "wL2P":"wholesale_level_2_price",
    "actionPrice":"action_price_qr",
    "aP":"action_price_qr",
    "actionCode":"action_code_qr",
    "aC":"action_code_qr",
}

def enhance_crop(img):
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    if max(h, w) < 700:
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    return img


def run_ocr(crop):
    crop = enhance_crop(crop)

    ocr = reader.predict(crop)
    if not ocr:
        return ""

    texts = ocr[0].get("rec_texts", [])
    return " ".join(texts)


def parse_qr(text):
    result = {v: "нет" for v in QR_KEYS.values()}

    for part in text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
        elif ":" in part:
            k, v = part.split(":", 1)
        else:
            continue

        k = k.strip()
        v = v.strip()

        if k in QR_KEYS:
            result[QR_KEYS[k]] = v

    return result


def extract_prices(text):
    prices = PRICE_RE.findall(text)
    prices = [p.replace(",", ".") for p in prices]

    return {
        "price_default": prices[0] if len(prices) > 0 else "нет",
        "price_card": prices[1] if len(prices) > 1 else "нет",
        "price_discount": prices[2] if len(prices) > 2 else "нет",
    }


def extract_barcode(text):
    found = BARCODE_RE.findall(text)
    return found[0] if found else "нет"


def detect_color(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean = hsv[:, :, 0].mean()

    if mean < 20:
        return "red"
    elif mean < 40:
        return "yellow"
    return "white"


def process_video(video_path):
    rows = []

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(fps * SAMPLE_EVERY_SEC), 1)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        timestamp = int(frame_idx / fps * 1000)

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        results = model.predict(frame, conf=0.25, verbose=False, device=0)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for x1, y1, x2, y2 in boxes.astype(int):
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                text = run_ocr(crop)

                qr_data = {}
                decoded = decode(crop)
                if decoded:
                    try:
                        qr_data = parse_qr(decoded[0].data.decode("utf-8"))
                    except:
                        pass

                prices = extract_prices(text)

                rows.append({
                    "filename": video_path.name,
                    "frame_timestamp": timestamp,
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                    "barcode": extract_barcode(text),
                    "product_name": text[:80],
                    "additional_info": text,
                    "color": detect_color(crop),
                    **prices,
                    **qr_data
                })

        frame_idx += 1

    cap.release()
    return rows


all_rows = []

videos = list(VIDEO_DIR.glob("*.mp4"))

for video in tqdm(videos):
    try:
        all_rows.extend(process_video(video))
    except Exception as e:
        print(video.name, e)

df = pd.DataFrame(all_rows)
df = df.reindex(columns=COLUMNS, fill_value="нет")

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Готово:", OUTPUT_CSV)