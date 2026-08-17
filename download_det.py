from datasets import load_dataset
from pathlib import Path
import json


NUM_TRAIN = 1000
NUM_VAL = 200

OUT = Path("data")
IMG_DIR = OUT / "det_images"

OUT.mkdir(exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


print("Loading CaptionedSynthText streaming...")

ds = load_dataset(
    "wendlerc/CaptionedSynthText",
    split="train",
    streaming=True,
)


rows = []

for i, sample in enumerate(ds):

    if len(rows) >= NUM_TRAIN + NUM_VAL:
        break

    try:

        # 실제 schema
        image = sample["jpg"]
        metadata = sample["json"]

        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        ann = metadata["ocr_annotation"]

        boxes = ann["bounding_boxes"]
        texts = ann["text"]

        if len(boxes) == 0:
            continue

        if image.mode != "RGB":
            image = image.convert("RGB")

        name = f"{len(rows):06d}.jpg"

        path = IMG_DIR / name

        image.save(
            path,
            quality=90
        )

        rows.append({
            "image": str(path),
            "boxes": boxes,
            "texts": texts,
        })

        if len(rows) % 100 == 0:

            print(
                "saved:",
                len(rows)
            )

    except Exception as e:

        print(
            "skip:",
            i,
            e
        )


print(
    "total:",
    len(rows)
)


# --------------------------------------
# DBNet annotation
# --------------------------------------

with open(
    OUT / "annotations.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        rows,
        f,
        ensure_ascii=False
    )


print()
print("========================")
print("images :", len(rows))
print("output : data/annotations.json")
print("========================")

print()
print("example:")
print(
    json.dumps(
        rows[0],
        ensure_ascii=False,
        indent=2
    )[:2000]
)
