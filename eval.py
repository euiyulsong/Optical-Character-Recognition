import json
import re
from pathlib import Path

import torch
from PIL import Image
from jiwer import cer
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

MODEL = "opendatalab/MinerU2.5-Pro-2604-1.2B"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    MODEL,
    use_fast=True,
)


def normalize(s):
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def predict(image_path):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": (
                        "Recognize all text in this image. "
                        "Output only the recognized text."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated = generated[
        :,
        inputs["input_ids"].shape[1]:
    ]

    pred = processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0]

    return normalize(pred)


samples = []

with open(
    "data/val_raw.jsonl",
    encoding="utf-8"
) as f:
    for line in f:
        samples.append(json.loads(line))


gts = []
preds = []

for i, x in enumerate(samples):

    gt = normalize(x["text"])
    pred = predict(x["image"])

    gts.append(gt)
    preds.append(pred)

    print(
        f"[{i:03d}]",
        "\nGT  :", gt,
        "\nPRED:", pred,
        "\n"
    )


score = cer(
    gts,
    preds,
)

exact = sum(
    g == p
    for g, p in zip(gts, preds)
) / len(gts)

print("============================")
print("samples     :", len(gts))
print("CER         :", score)
print("Exact Match :", exact)
print("============================")
