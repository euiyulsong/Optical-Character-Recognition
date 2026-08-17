import json
from pathlib import Path


def convert(src, dst):
    output = []

    with open(src, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            image = x["image"]
            text = x["text"]

            item = {
                "messages": [
                    {
                        "role": "user",
                        "content": "<image>Recognize all text in this image. Output only the recognized text."
                    },
                    {
                        "role": "assistant",
                        "content": text
                    }
                ],
                "images": [image]
            }

            output.append(item)

    with open(dst, "w", encoding="utf-8") as f:
        for x in output:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False
                ) + "\n"
            )

    print(dst, len(output))


convert(
    "data/train_raw.jsonl",
    "data/train.jsonl"
)

convert(
    "data/val_raw.jsonl",
    "data/val.jsonl"
)
