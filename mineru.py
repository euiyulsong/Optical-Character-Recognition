# mineru.py

import os
import gc
import re
import json
import math
from contextlib import nullcontext

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    get_linear_schedule_with_warmup,
)


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "opendatalab/MinerU2.5-2509-1.2B"

TRAIN_FILE = "data/train_raw.jsonl"
VAL_FILE = "data/val_raw.jsonl"

OUTPUT_DIR = "./mineru25_sft"

RESULT_DIR = "results"
RESULT_FILE = os.path.join(
    RESULT_DIR,
    "mineru25.json",
)

EPOCHS = 1
LR = 2e-5

BATCH_SIZE = 16
GRAD_ACCUM = 8

MAX_TRAIN_SAMPLES = 1000
MAX_VAL_SAMPLES = 200

MAX_NEW_TOKENS = 32

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DTYPE = (
    torch.bfloat16
    if DEVICE == "cuda"
    else torch.float32
)


# ============================================================
# CUDA
# ============================================================

# 네 환경에서 cuDNN initialization 문제가 있었으므로
# CUDA는 쓰되 cuDNN만 끈다.
torch.backends.cudnn.enabled = False

print("CUDA:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)


# ============================================================
# PROMPT
# ============================================================

# download.py의 recognition crop 기준
PROMPT = (
    "Recognize all text in this image. "
    "Output only the recognized text."
)


# ============================================================
# DATASET
# ============================================================

class OCRDataset(Dataset):

    def __init__(
        self,
        jsonl_path,
        max_samples=None,
    ):

        self.rows = []

        with open(
            jsonl_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                row = json.loads(line)

                image = row.get("image")
                text = row.get("text")

                if image is None:
                    continue

                if text is None:
                    continue

                if not os.path.exists(image):

                    print(
                        "missing image:",
                        image,
                    )

                    continue

                text = str(text).strip()

                if not text:
                    continue

                self.rows.append({
                    "image": image,
                    "text": text,
                })

        if max_samples is not None:

            self.rows = (
                self.rows[:max_samples]
            )

        print(
            f"{jsonl_path}: "
            f"{len(self.rows)} samples"
        )

    def __len__(self):

        return len(self.rows)

    def __getitem__(
        self,
        idx,
    ):

        return self.rows[idx]


# ============================================================
# MODEL LOADER
# ============================================================

def load_model(
    model_path,
    training=False,
):

    print()
    print("=" * 70)
    print("LOAD MODEL:", model_path)
    print("=" * 70)

    processor = (
        AutoProcessor
        .from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
    )

    model = (
        AutoModelForMultimodalLM
        .from_pretrained(
            model_path,
            dtype=DTYPE,
            trust_remote_code=True,
        )
    )

    print(
        "loaded class:",
        model.__class__.__name__,
    )

    print(
        "tie_word_embeddings:",
        getattr(
            model.config,
            "tie_word_embeddings",
            None,
        ),
    )

    model.to(DEVICE)

    if training:

        model.train()

        if hasattr(
            model,
            "gradient_checkpointing_enable",
        ):

            model.gradient_checkpointing_enable()

        if hasattr(
            model.config,
            "use_cache",
        ):

            model.config.use_cache = False

    else:

        model.eval()

        if hasattr(
            model.config,
            "use_cache",
        ):

            model.config.use_cache = True

    return model, processor


# ============================================================
# TRAIN COLLATOR
# ============================================================

class OCRTrainCollator:

    def __init__(
        self,
        processor,
    ):

        self.processor = processor

    def __call__(
        self,
        batch,
    ):

        # 이 quickstart는 batch=1
        row = batch[0]

        image = (
            Image
            .open(row["image"])
            .convert("RGB")
        )

        answer = row["text"]

        # --------------------------------------------------
        # Prompt only
        # --------------------------------------------------

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ]

        prompt_inputs = (
            self.processor
            .apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        prompt_len = (
            prompt_inputs[
                "input_ids"
            ].shape[1]
        )

        # --------------------------------------------------
        # Full SFT sequence
        # --------------------------------------------------

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
                        "text": PROMPT,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    },
                ],
            },
        ]

        inputs = (
            self.processor
            .apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
        )

        labels = (
            inputs[
                "input_ids"
            ].clone()
        )

        # --------------------------------------------------
        # User prompt는 loss 제외
        # assistant answer만 학습
        # --------------------------------------------------

        prompt_len = min(
            prompt_len,
            labels.shape[1],
        )

        labels[
            :,
            :prompt_len,
        ] = -100

        # padding loss 제외

        pad_id = (
            self.processor
            .tokenizer
            .pad_token_id
        )

        if pad_id is not None:

            labels[
                inputs["input_ids"]
                == pad_id
            ] = -100

        inputs["labels"] = labels

        return inputs


# ============================================================
# METRICS
# ============================================================

def normalize_text(
    x,
):

    x = str(x)

    x = re.sub(
        r"\s+",
        " ",
        x,
    )

    return x.strip()


def exact_match(
    pred,
    gt,
):

    return float(
        normalize_text(pred)
        ==
        normalize_text(gt)
    )


def levenshtein(
    a,
    b,
):

    prev = list(
        range(
            len(b) + 1
        )
    )

    for i, ca in enumerate(
        a,
        start=1,
    ):

        cur = [i]

        for j, cb in enumerate(
            b,
            start=1,
        ):

            insert_cost = (
                cur[j - 1] + 1
            )

            delete_cost = (
                prev[j] + 1
            )

            replace_cost = (
                prev[j - 1]
                +
                (
                    0
                    if ca == cb
                    else 1
                )
            )

            cur.append(
                min(
                    insert_cost,
                    delete_cost,
                    replace_cost,
                )
            )

        prev = cur

    return prev[-1]


def cer(
    pred,
    gt,
):

    pred = normalize_text(pred)
    gt = normalize_text(gt)

    if not gt:

        return (
            0.0
            if not pred
            else 1.0
        )

    return (
        levenshtein(
            pred,
            gt,
        )
        / len(gt)
    )


# ============================================================
# TRAIN
# ============================================================

def train():

    model, processor = (
        load_model(
            BASE_MODEL,
            training=True,
        )
    )

    train_dataset = (
        OCRDataset(
            TRAIN_FILE,
            MAX_TRAIN_SAMPLES,
        )
    )

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=OCRTrainCollator(
                processor
            ),
        )
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=0.01,
        )
    )

    update_steps_per_epoch = (
        math.ceil(
            len(train_loader)
            / GRAD_ACCUM
        )
    )

    total_update_steps = (
        update_steps_per_epoch
        * EPOCHS
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(
                1,
                int(
                    total_update_steps
                    * 0.05
                ),
            ),
            num_training_steps=max(
                1,
                total_update_steps,
            ),
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    print()
    print("=" * 70)
    print("TRAIN")
    print("=" * 70)

    for epoch in range(EPOCHS):

        total_loss = 0.0

        for step, batch in enumerate(
            train_loader
        ):

            batch = {
                k: (
                    v.to(DEVICE)
                    if torch.is_tensor(v)
                    else v
                )
                for k, v
                in batch.items()
            }

            # --------------------------------------------------
            # First batch debug
            # --------------------------------------------------

            if (
                epoch == 0
                and step == 0
            ):

                print(
                    "input_ids:",
                    tuple(
                        batch[
                            "input_ids"
                        ].shape
                    ),
                )

                if (
                    "pixel_values"
                    in batch
                ):

                    print(
                        "pixel_values:",
                        tuple(
                            batch[
                                "pixel_values"
                            ].shape
                        ),
                    )

                if (
                    "image_grid_thw"
                    in batch
                ):

                    print(
                        "image_grid_thw:",
                        batch[
                            "image_grid_thw"
                        ],
                    )

                answer_tokens = (
                    batch["labels"]
                    != -100
                ).sum().item()

                print(
                    "trainable answer tokens:",
                    answer_tokens,
                )

            ctx = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                )
                if DEVICE == "cuda"
                else nullcontext()
            )

            with ctx:

                outputs = model(
                    **batch
                )

                raw_loss = (
                    outputs.loss
                )

                loss = (
                    raw_loss
                    / GRAD_ACCUM
                )

            loss.backward()

            total_loss += (
                raw_loss.item()
            )

            do_update = (
                (
                    (step + 1)
                    % GRAD_ACCUM
                    == 0
                )
                or
                (
                    step + 1
                    == len(train_loader)
                )
            )

            if do_update:

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            if step % 10 == 0:

                print(
                    f"epoch={epoch+1}/{EPOCHS} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={raw_loss.item():.4f}"
                )

        avg_loss = (
            total_loss
            / max(
                1,
                len(train_loader),
            )
        )

        print(
            f"epoch={epoch+1}/{EPOCHS} "
            f"avg_loss={avg_loss:.4f}"
        )

    print()
    print("=" * 70)
    print("SAVE")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    if hasattr(
        model.config,
        "use_cache",
    ):

        model.config.use_cache = True

    model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
    )

    processor.save_pretrained(
        OUTPUT_DIR
    )

    print(
        "saved:",
        OUTPUT_DIR,
    )

    del model
    del processor
    del train_loader
    del train_dataset

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# GENERATE
# ============================================================

@torch.inference_mode()
def generate_text(
    model,
    processor,
    image_path,
):

    image = (
        Image
        .open(image_path)
        .convert("RGB")
    )

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
                    "text": PROMPT,
                },
            ],
        }
    ]

    inputs = (
        processor
        .apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    )

    inputs = {
        k: (
            v.to(DEVICE)
            if torch.is_tensor(v)
            else v
        )
        for k, v
        in inputs.items()
    }

    prompt_len = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    output_ids = (
        model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    )

    generated = (
        output_ids[
            :,
            prompt_len:
        ]
    )

    text = (
        processor
        .batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )

    return text.strip()


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    model_path,
    rows,
    name,
):

    model, processor = (
        load_model(
            model_path,
            training=False,
        )
    )

    print()
    print("=" * 70)
    print("EVALUATE:", name)
    print("=" * 70)

    em_list = []
    cer_list = []

    predictions = []

    for i, row in enumerate(
        rows
    ):

        pred = generate_text(
            model,
            processor,
            row["image"],
        )

        gt = row["text"]

        em_score = (
            exact_match(
                pred,
                gt,
            )
        )

        cer_score = (
            cer(
                pred,
                gt,
            )
        )

        em_list.append(
            em_score
        )

        cer_list.append(
            cer_score
        )

        predictions.append({
            "image": row["image"],
            "ground_truth": gt,
            "prediction": pred,
            "EM": em_score,
            "CER": cer_score,
        })

        print(
            f"[{i+1}/{len(rows)}] "
            f"GT={gt!r} "
            f"PRED={pred!r} "
            f"EM={em_score:.0f} "
            f"CER={cer_score:.4f}"
        )

    result = {
        "name": name,
        "samples": len(rows),
        "EM": (
            sum(em_list)
            / max(
                1,
                len(em_list),
            )
        ),
        "CER": (
            sum(cer_list)
            / max(
                1,
                len(cer_list),
            )
        ),
        "predictions": predictions,
    }

    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return result


# ============================================================
# PRINT TABLE
# ============================================================

def print_table(
    base,
    ft,
):

    print()
    print("=" * 75)

    print(
        f"{'Model':<32}"
        f"{'Samples':>10}"
        f"{'EM ↑':>14}"
        f"{'CER ↓':>14}"
    )

    print("-" * 75)

    print(
        f"{'MinerU2.5 Base':<32}"
        f"{base['samples']:>10}"
        f"{base['EM']:>14.4f}"
        f"{base['CER']:>14.4f}"
    )

    print(
        f"{'MinerU2.5 Fine-tuned':<32}"
        f"{ft['samples']:>10}"
        f"{ft['EM']:>14.4f}"
        f"{ft['CER']:>14.4f}"
    )

    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("MinerU2.5 OCR TRAIN + EVALUATION")
    print("=" * 70)

    print(
        "device:",
        DEVICE,
    )

    print(
        "base model:",
        BASE_MODEL,
    )

    # ======================================================
    # 1. TRAIN
    # ======================================================

    train()

    # ======================================================
    # 2. VALIDATION
    # ======================================================

    val_dataset = (
        OCRDataset(
            VAL_FILE,
            MAX_VAL_SAMPLES,
        )
    )

    rows = [
        val_dataset[i]
        for i
        in range(
            len(val_dataset)
        )
    ]

    if len(rows) == 0:

        raise RuntimeError(
            "Validation dataset is empty."
        )

    # ======================================================
    # 3. BASE MODEL
    # ======================================================

    base_result = (
        evaluate(
            BASE_MODEL,
            rows,
            "MinerU2.5 Base",
        )
    )

    # ======================================================
    # 4. FINETUNED MODEL
    # ======================================================

    ft_result = (
        evaluate(
            OUTPUT_DIR,
            rows,
            "MinerU2.5 Fine-tuned",
        )
    )

    # ======================================================
    # 5. TABLE
    # ======================================================

    print_table(
        base_result,
        ft_result,
    )

    # ======================================================
    # 6. SAVE RESULTS
    # ======================================================

    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "base": base_result,
                "finetuned": ft_result,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "saved:",
        RESULT_FILE,
    )
