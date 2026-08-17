from datasets import load_dataset
from pathlib import Path
import json
import random

NUM_TRAIN = 1000
NUM_VAL = 200
NUM_TOTAL = NUM_TRAIN + NUM_VAL

OUT = Path("data")
IMG_DIR = OUT / "rec_images"

OUT.mkdir(exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading CaptionedSynthText in streaming mode...")

ds = load_dataset(
    "wendlerc/CaptionedSynthText",
    split="train",
    streaming=True,
)

rows = []

for i, sample in enumerate(ds):
    if len(rows) >= NUM_TOTAL:
        break

    try:
        # 실제 column 이름
        image = sample["jpg"]

        ann = sample["json"]["ocr_annotation"]

        texts = ann["text"]
        boxes = ann["bounding_boxes"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        for text, box in zip(texts, boxes):

            if len(rows) >= NUM_TOTAL:
                break

            text = str(text).strip()

            if not text:
                continue

            # polygon 4 points
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]

            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(image.width, int(max(xs)))
            y2 = min(image.height, int(max(ys)))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))

            # 너무 작은 crop 제거
            if crop.width < 4 or crop.height < 4:
                continue

            img_name = f"{len(rows):06d}.jpg"
            img_path = IMG_DIR / img_name

            crop.save(img_path, quality=90)

            rows.append({
                "image": str(img_path.resolve()),
                "text": text,
            })

            if len(rows) % 100 == 0:
                print("generated:", len(rows))

    except Exception as e:
        print("skip:", i, repr(e))


print("total:", len(rows))

random.Random(42).shuffle(rows)

train = rows[:NUM_TRAIN]
val = rows[NUM_TRAIN:NUM_TRAIN + NUM_VAL]


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )


write_jsonl(
    OUT / "train_raw.jsonl",
    train,
)

write_jsonl(
    OUT / "val_raw.jsonl",
    val,
)

print()
print("==============================")
print("train :", len(train))
print("val   :", len(val))
print("images:", IMG_DIR)
print("==============================")
