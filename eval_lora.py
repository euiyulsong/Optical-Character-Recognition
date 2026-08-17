import json
import re

import torch
from PIL import Image
from jiwer import cer
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
from peft import PeftModel


BASE_MODEL = "opendatalab/MinerU2.5-Pro-2604-1.2B"

# 실제 생성된 checkpoint로 변경
LORA_PATH = "output/mineru_lora/v1-20260814-115722/checkpoint-125/"


# ==========================================
# 1. Base model
# ==========================================

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ==========================================
# 2. LoRA adapter
# ==========================================

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

model.eval()

print("LoRA loaded:", LORA_PATH)


# processor는 base model 것 그대로 사용
processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    use_fast=True,
)


def normalize(s):
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

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

    # device_map=auto를 쓴 모델에서는
    # embedding이 올라간 device로 input을 보내는 게 안전
    device = next(model.parameters()).device

    inputs = {
        k: v.to(device)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # prompt 부분 제거
    generated = generated[
        :,
        inputs["input_ids"].shape[1]:
    ]

    pred = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return normalize(pred)


# ==========================================
# validation set
# ==========================================

samples = []

with open(
    "data/val_raw.jsonl",
    encoding="utf-8",
) as f:

    for line in f:
        samples.append(
            json.loads(line)
        )


gts = []
preds = []


for i, x in enumerate(samples):

    gt = normalize(
        x["text"]
    )

    pred = predict(
        x["image"]
    )

    gts.append(gt)
    preds.append(pred)

    print(
        f"[{i:03d}]"
        f"\nGT   : {gt}"
        f"\nPRED : {pred}"
        "\n"
    )


# ==========================================
# metrics
# ==========================================

score = cer(
    gts,
    preds,
)

exact = sum(
    g == p
    for g, p in zip(gts, preds)
) / len(gts)


print("=" * 40)

print(
    "samples     :",
    len(gts)
)

print(
    "CER         :",
    score
)

print(
    "Exact Match :",
    exact
)

print("=" * 40)
