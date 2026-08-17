import os
import re
import gc
import json
import random
import string
import argparse
from io import BytesIO
from collections import defaultdict, Counter

import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image

from tqdm import tqdm

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from rank_bm25 import BM25Okapi

from colpali_engine.models import (
    ColPali,
    ColPaliProcessor,
)
import torch

torch.backends.cudnn.enabled = False
print("CUDA:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)

# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "pixparse/docvqa-single-page-questions"

DENSE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

COLPALI_MODEL_NAME = "vidore/colpali-v1.3"

VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

OUTPUT_DIR = "./outputs"

DENSE_SAVE_DIR = os.path.join(
    OUTPUT_DIR,
    "dense_ft",
)

COLPALI_SAVE_DIR = os.path.join(
    OUTPUT_DIR,
    "colpali_ft",
)


if torch.cuda.is_available():
    DEVICE = "cuda"
    COLPALI_DEVICE_MAP = "cuda:0"
else:
    DEVICE = "cpu"
    COLPALI_DEVICE_MAP = "cpu"


if (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
):
    MODEL_DTYPE = torch.bfloat16
else:
    MODEL_DTYPE = torch.float32


SEED = 42


# ============================================================
# SEED
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# Dataset helper
# ============================================================

def get_page_id(row):

    meta = row["other_metadata"]

    if isinstance(meta, dict):

        if "image" in meta:
            return str(
                meta["image"]
            )

        if "doc_id" in meta:

            page = meta.get(
                "ucsf_document_page_no",
                "",
            )

            return (
                f"{meta['doc_id']}_"
                f"{page}"
            )

    return str(
        row["question_id"]
    )


def get_pil_image(row):

    image = row["image"]

    if isinstance(
        image,
        Image.Image,
    ):

        return image.convert(
            "RGB"
        )

    if isinstance(
        image,
        dict,
    ):

        if image.get(
            "bytes"
        ) is not None:

            return Image.open(
                BytesIO(
                    image["bytes"]
                )
            ).convert(
                "RGB"
            )

        if image.get(
            "path"
        ):

            return Image.open(
                image["path"]
            ).convert(
                "RGB"
            )

    if isinstance(
        image,
        bytes,
    ):

        return Image.open(
            BytesIO(image)
        ).convert(
            "RGB"
        )

    raise TypeError(
        f"Unsupported image type: "
        f"{type(image)}"
    )


def extract_ocr(row):

    ocr = row.get(
        "ocr_results",
        {},
    )

    if ocr is None:
        return ""

    if isinstance(
        ocr,
        str,
    ):
        return ocr

    texts = []

    def recursive_extract(obj):

        if isinstance(
            obj,
            dict,
        ):

            if (
                "text" in obj
                and isinstance(
                    obj["text"],
                    str,
                )
            ):

                text = (
                    obj["text"]
                    .strip()
                )

                if text:
                    texts.append(
                        text
                    )

            for value in (
                obj.values()
            ):

                recursive_extract(
                    value
                )

        elif isinstance(
            obj,
            list,
        ):

            for item in obj:

                recursive_extract(
                    item
                )

    recursive_extract(
        ocr
    )

    # Duplicate OCR fields can occur.
    deduped = []

    seen = set()

    for text in texts:

        if text not in seen:

            seen.add(text)

            deduped.append(
                text
            )

    return "\n".join(
        deduped
    )


# ============================================================
# EM / F1
# ============================================================

def normalize_answer(text):

    if text is None:
        return ""

    text = (
        str(text)
        .lower()
    )

    text = "".join(
        c
        for c in text
        if c
        not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    text = " ".join(
        text.split()
    )

    return text


def exact_match(
    prediction,
    answers,
):

    pred = normalize_answer(
        prediction
    )

    return max(
        (
            float(
                pred
                ==
                normalize_answer(
                    answer
                )
            )
            for answer
            in answers
        ),
        default=0.0,
    )


def f1_single(
    prediction,
    answer,
):

    pred_tokens = (
        normalize_answer(
            prediction
        )
        .split()
    )

    answer_tokens = (
        normalize_answer(
            answer
        )
        .split()
    )

    if not pred_tokens:

        return float(
            not answer_tokens
        )

    if not answer_tokens:

        return 0.0

    common = (
        Counter(
            pred_tokens
        )
        &
        Counter(
            answer_tokens
        )
    )

    overlap = sum(
        common.values()
    )

    if overlap == 0:
        return 0.0

    precision = (
        overlap
        / len(pred_tokens)
    )

    recall = (
        overlap
        / len(answer_tokens)
    )

    return (
        2
        * precision
        * recall
        /
        (
            precision
            + recall
        )
    )


def token_f1(
    prediction,
    answers,
):

    return max(
        (
            f1_single(
                prediction,
                answer,
            )
            for answer
            in answers
        ),
        default=0.0,
    )


# ============================================================
# DATA
# ============================================================

def load_docvqa(
    train_limit,
    val_limit,
):

    print(
        "\nLoading DocVQA..."
    )

    train = load_dataset(
        DATASET_NAME,
        split="train",
    )

    val = load_dataset(
        DATASET_NAME,
        split="validation",
    )

    if train_limit:

        train = train.select(
            range(
                min(
                    train_limit,
                    len(train),
                )
            )
        )

    if val_limit:

        val = val.select(
            range(
                min(
                    val_limit,
                    len(val),
                )
            )
        )

    print(
        "train:",
        len(train),
    )

    print(
        "validation:",
        len(val),
    )

    return train, val


def build_corpus(dataset):

    page_map = {}

    queries = []

    print(
        "\nBuilding page corpus..."
    )

    for row in tqdm(
        dataset
    ):

        pid = get_page_id(
            row
        )

        if (
            pid
            not in page_map
        ):

            page_map[
                pid
            ] = {
                "page_id":
                    pid,

                "image":
                    get_pil_image(
                        row
                    ),

                "ocr":
                    extract_ocr(
                        row
                    ),
            }

        queries.append({
            "question":
                row["question"],

            "answers":
                row["answers"],

            "page_id":
                pid,
        })

    corpus = list(
        page_map.values()
    )

    print(
        "unique pages:",
        len(corpus),
    )

    print(
        "queries:",
        len(queries),
    )

    return corpus, queries


# ============================================================
# Unique-page batches
# ============================================================

def build_page_groups(
    dataset,
):

    groups = (
        defaultdict(list)
    )

    for i in range(
        len(dataset)
    ):

        pid = get_page_id(
            dataset[i]
        )

        groups[pid].append(
            i
        )

    return groups


def get_unique_page_rows(
    dataset,
    groups,
):

    page_ids = list(
        groups.keys()
    )

    random.shuffle(
        page_ids
    )

    rows = []

    for pid in page_ids:

        idx = random.choice(
            groups[pid]
        )

        rows.append(
            dataset[idx]
        )

    return rows


# ============================================================
# Retrieval metrics
# ============================================================

def recall_at_k(
    rankings,
    queries,
    corpus,
    k,
):

    hits = 0

    for query, rank in zip(
        queries,
        rankings,
    ):

        found = False

        for idx in rank[:k]:

            if (
                corpus[
                    int(idx)
                ]["page_id"]
                ==
                query[
                    "page_id"
                ]
            ):

                found = True
                break

        hits += int(
            found
        )

    return (
        hits
        / len(queries)
    )


def mean_reciprocal_rank(
    rankings,
    queries,
    corpus,
):

    score = 0.0

    for query, rank in zip(
        queries,
        rankings,
    ):

        for pos, idx in enumerate(
            rank
        ):

            if (
                corpus[
                    int(idx)
                ]["page_id"]
                ==
                query[
                    "page_id"
                ]
            ):

                score += (
                    1.0
                    /
                    (pos + 1)
                )

                break

    return (
        score
        / len(queries)
    )


def retrieval_metrics(
    rankings,
    queries,
    corpus,
):

    return {
        "Recall@1":
            recall_at_k(
                rankings,
                queries,
                corpus,
                1,
            ),

        "Recall@5":
            recall_at_k(
                rankings,
                queries,
                corpus,
                5,
            ),

        "MRR":
            mean_reciprocal_rank(
                rankings,
                queries,
                corpus,
            ),
    }


# ============================================================
# BM25
# ============================================================

def tokenize_bm25(text):

    return re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )


def run_bm25(
    corpus,
    queries,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OCR + BM25"
    )

    print(
        "=" * 80
    )

    documents = [
        tokenize_bm25(
            x["ocr"]
        )
        for x in corpus
    ]

    bm25 = BM25Okapi(
        documents
    )

    rankings = []

    for query in tqdm(
        queries
    ):

        scores = (
            bm25.get_scores(
                tokenize_bm25(
                    query[
                        "question"
                    ]
                )
            )
        )

        rank = np.argsort(
            -np.asarray(
                scores
            )
        )

        rankings.append(
            rank
        )

    return np.asarray(
        rankings
    )


# ============================================================
# DENSE RETRIEVER
# ============================================================

class DenseRetriever:

    def __init__(
        self,
        model_name,
        max_length=512,
    ):

        print(
            "\nLoading Dense:",
            model_name,
        )

        self.max_length = (
            max_length
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModel
            .from_pretrained(
                model_name
            )
            .to(DEVICE)
        )


    @staticmethod
    def mean_pool(
        outputs,
        attention_mask,
    ):

        hidden = (
            outputs
            .last_hidden_state
        )

        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand_as(hidden)
            .float()
        )

        summed = (
            hidden
            * mask
        ).sum(
            dim=1
        )

        counts = (
            mask
            .sum(dim=1)
            .clamp(
                min=1e-9
            )
        )

        return (
            summed
            / counts
        )


    def encode_train(
        self,
        texts,
        is_query,
    ):

        if is_query:

            texts = [
                (
                    "Represent this sentence "
                    "for searching relevant "
                    "passages: "
                    + text
                )
                for text
                in texts
            ]

        batch = (
            self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=(
                    self.max_length
                ),
                return_tensors="pt",
            )
        )

        batch = {
            key: value.to(
                DEVICE
            )
            for key, value
            in batch.items()
        }

        output = (
            self.model(
                **batch
            )
        )

        embeddings = (
            self.mean_pool(
                output,
                batch[
                    "attention_mask"
                ],
            )
        )

        embeddings = (
            F.normalize(
                embeddings,
                p=2,
                dim=-1,
            )
        )

        return embeddings


    def train(
        self,
        train_dataset,
        batch_size,
        max_steps,
        lr=2e-5,
        temperature=0.05,
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            "TRAIN OCR + DENSE"
        )

        print(
            "=" * 80
        )

        groups = (
            build_page_groups(
                train_dataset
            )
        )

        optimizer = (
            torch.optim.AdamW(
                self.model
                .parameters(),
                lr=lr,
            )
        )

        self.model.train()

        step = 0

        while (
            step
            < max_steps
        ):

            rows = (
                get_unique_page_rows(
                    train_dataset,
                    groups,
                )
            )

            for start in range(
                0,
                len(rows),
                batch_size,
            ):

                batch_rows = rows[
                    start:
                    start
                    + batch_size
                ]

                if (
                    len(batch_rows)
                    < 2
                ):
                    continue

                questions = [
                    row[
                        "question"
                    ]
                    for row
                    in batch_rows
                ]

                docs = [
                    extract_ocr(
                        row
                    )
                    for row
                    in batch_rows
                ]

                q_emb = (
                    self.encode_train(
                        questions,
                        True,
                    )
                )

                d_emb = (
                    self.encode_train(
                        docs,
                        False,
                    )
                )

                scores = (
                    q_emb
                    @ d_emb.T
                ) / temperature

                labels = (
                    torch.arange(
                        len(
                            batch_rows
                        ),
                        device=DEVICE,
                    )
                )

                loss = (
                    F.cross_entropy(
                        scores,
                        labels,
                    )
                    +
                    F.cross_entropy(
                        scores.T,
                        labels,
                    )
                ) / 2

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model
                    .parameters(),
                    1.0,
                )

                optimizer.step()

                if (
                    step % 20
                    == 0
                ):

                    acc = (
                        (
                            scores.argmax(
                                dim=1
                            )
                            == labels
                        )
                        .float()
                        .mean()
                        .item()
                    )

                    print(
                        f"dense "
                        f"step={step} "
                        f"loss="
                        f"{loss.item():.4f} "
                        f"acc="
                        f"{acc:.4f}"
                    )

                step += 1

                if (
                    step
                    >= max_steps
                ):
                    break


    def save(
        self,
        path,
    ):

        os.makedirs(
            path,
            exist_ok=True,
        )

        self.model.save_pretrained(
            path
        )

        self.tokenizer.save_pretrained(
            path
        )

        print(
            "Saved dense model:",
            path,
        )


    @torch.inference_mode()
    def encode_eval(
        self,
        texts,
        is_query,
        batch_size=32,
    ):

        self.model.eval()

        outputs = []

        for start in tqdm(
            range(
                0,
                len(texts),
                batch_size,
            )
        ):

            emb = (
                self.encode_train(
                    texts[
                        start:
                        start
                        + batch_size
                    ],
                    is_query,
                )
            )

            outputs.append(
                emb.cpu()
            )

        return torch.cat(
            outputs,
            dim=0,
        )


    def retrieve(
        self,
        corpus,
        queries,
    ):

        print(
            "\nDense corpus encoding..."
        )

        docs = [
            x["ocr"]
            for x in corpus
        ]

        d_emb = (
            self.encode_eval(
                docs,
                False,
            )
        )

        print(
            "\nDense query encoding..."
        )

        question_texts = [
            x["question"]
            for x in queries
        ]

        q_emb = (
            self.encode_eval(
                question_texts,
                True,
            )
        )

        scores = (
            q_emb
            @ d_emb.T
        )

        rankings = (
            torch.argsort(
                scores,
                dim=1,
                descending=True,
            )
            .numpy()
        )

        return rankings


# ============================================================
# COLPALI
# ============================================================

class ColPaliRetriever:

    def __init__(
        self,
        model_name,
    ):

        print(
            "\nLoading ColPali:",
            model_name,
        )

        self.processor = (
            ColPaliProcessor
            .from_pretrained(
                model_name
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # DO NOT:
        #
        # ColPali.from_pretrained(...).to("cuda")
        #
        # Load directly onto the target device.
        # ----------------------------------------------------

        self.model = (
            ColPali
            .from_pretrained(
                model_name,
                torch_dtype=(
                    MODEL_DTYPE
                ),
                device_map=(
                    COLPALI_DEVICE_MAP
                ),
            )
        )

        print(
            "ColPali device:",
            self.model.device,
        )


    @property
    def device(self):

        # Official ColPali examples expose model.device.
        return self.model.device


    def prepare_for_training(self):

        print("\nEnabling LoRA-only training...")

        for param in self.model.parameters():
            param.requires_grad = False

        trainable_names = []
        trainable_count = 0

        for name, param in self.model.named_parameters():

            if (
                "lora_A" in name
                or "lora_B" in name
            ):
                param.requires_grad = True

                trainable_names.append(name)
                trainable_count += param.numel()

        if trainable_count == 0:
            raise RuntimeError(
                "No LoRA parameters found."
            )

        print("\nTrainable LoRA params:")

        for name in trainable_names:
            print(" ", name)

        total = sum(
            p.numel()
            for p in self.model.parameters()
        )

        print(
            f"\nTrainable: {trainable_count:,}"
        )

        print(
            f"Total: {total:,}"
        )

        print(
            f"Ratio: "
            f"{trainable_count / total * 100:.4f}%"
        )

    def train(
        self,
        train_dataset,
        batch_size,
        max_steps,
        grad_accum,
        lr=5e-5,
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            "TRAIN COLPALI"
        )

        print(
            "=" * 80
        )

        self.prepare_for_training()

        groups = (
            build_page_groups(
                train_dataset
            )
        )

        params = [
            parameter
            for parameter in
            self.model.parameters()
            if (
                parameter
                .requires_grad
            )
        ]

        optimizer = (
            torch.optim.AdamW(
                params,
                lr=lr,
            )
        )

        self.model.train()

        if hasattr(
            self.model.config,
            "use_cache",
        ):

            self.model.config.use_cache = (
                False
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        optimizer_step = 0

        micro_step = 0

        while (
            optimizer_step
            < max_steps
        ):

            rows = (
                get_unique_page_rows(
                    train_dataset,
                    groups,
                )
            )

            for start in range(
                0,
                len(rows),
                batch_size,
            ):

                batch_rows = rows[
                    start:
                    start
                    + batch_size
                ]

                if (
                    len(batch_rows)
                    < 2
                ):
                    continue

                questions = [
                    row[
                        "question"
                    ]
                    for row
                    in batch_rows
                ]

                images = [
                    get_pil_image(
                        row
                    )
                    for row
                    in batch_rows
                ]

                q_inputs = (
                    self.processor
                    .process_queries(
                        questions
                    )
                    .to(
                        self.device
                    )
                )

                d_inputs = (
                    self.processor
                    .process_images(
                        images
                    )
                    .to(
                        self.device
                    )
                )

                q_emb = (
                    self.model(
                        **q_inputs
                    )
                )

                d_emb = (
                    self.model(
                        **d_inputs
                    )
                )

                # --------------------------------------------
                # Use ColPali's official scoring path.
                #
                # It computes ColBERT-style MaxSim.
                # --------------------------------------------

                scores = (
                    self.processor
                    .score_multi_vector(
                        q_emb,
                        d_emb,
                    )
                )

                labels = (
                    torch.arange(
                        len(
                            batch_rows
                        ),
                        device=(
                            scores.device
                        ),
                    )
                )

                loss = (
                    F.cross_entropy(
                        scores,
                        labels,
                    )
                    +
                    F.cross_entropy(
                        scores.T,
                        labels,
                    )
                ) / 2

                raw_loss = (
                    loss.item()
                )

                (
                    loss
                    / grad_accum
                ).backward()

                micro_step += 1

                if (
                    micro_step
                    % grad_accum
                    == 0
                ):

                    torch.nn.utils.clip_grad_norm_(
                        params,
                        1.0,
                    )

                    optimizer.step()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    if (
                        optimizer_step
                        % 10
                        == 0
                    ):

                        acc = (
                            (
                                scores.argmax(
                                    dim=1
                                )
                                == labels
                            )
                            .float()
                            .mean()
                            .item()
                        )

                        print(
                            "colpali "
                            f"step="
                            f"{optimizer_step} "
                            f"loss="
                            f"{raw_loss:.4f} "
                            f"acc="
                            f"{acc:.4f}"
                        )

                    optimizer_step += (
                        1
                    )

                del (
                    q_emb,
                    d_emb,
                    scores,
                    loss,
                )

                if (
                    optimizer_step
                    >= max_steps
                ):
                    break


    def save(
        self,
        path,
    ):

        os.makedirs(
            path,
            exist_ok=True,
        )

        self.model.save_pretrained(
            path
        )

        self.processor.save_pretrained(
            path
        )

        print(
            "Saved ColPali:",
            path,
        )


    @torch.inference_mode()
    def encode_pages(
        self,
        corpus,
        batch_size=2,
    ):

        self.model.eval()

        embeddings = []

        print(
            "\nEncoding pages with ColPali..."
        )

        for start in tqdm(
            range(
                0,
                len(corpus),
                batch_size,
            )
        ):

            images = [
                x["image"]
                for x
                in corpus[
                    start:
                    start
                    + batch_size
                ]
            ]

            inputs = (
                self.processor
                .process_images(
                    images
                )
                .to(
                    self.device
                )
            )

            emb = (
                self.model(
                    **inputs
                )
            )

            for x in emb:

                embeddings.append(
                    x.detach()
                    .cpu()
                )

        return embeddings


    @torch.inference_mode()
    def encode_queries(
        self,
        queries,
        batch_size=8,
    ):

        self.model.eval()

        embeddings = []

        print(
            "\nEncoding queries with ColPali..."
        )

        for start in tqdm(
            range(
                0,
                len(queries),
                batch_size,
            )
        ):

            texts = [
                x["question"]
                for x
                in queries[
                    start:
                    start
                    + batch_size
                ]
            ]

            inputs = (
                self.processor
                .process_queries(
                    texts
                )
                .to(
                    self.device
                )
            )

            emb = (
                self.model(
                    **inputs
                )
            )

            for x in emb:

                embeddings.append(
                    x.detach()
                    .cpu()
                )

        return embeddings


    def retrieve(
        self,
        corpus,
        queries,
        doc_batch_size=16,
    ):

        page_embs = (
            self.encode_pages(
                corpus
            )
        )

        query_embs = (
            self.encode_queries(
                queries
            )
        )

        rankings = []

        print(
            "\nColPali MaxSim retrieval..."
        )

        # For ~1k pages brute force is fine.
        # score_multi_vector handles padding / MaxSim.

        for q in tqdm(
            query_embs
        ):

            score_parts = []

            q = q.unsqueeze(
                0
            )

            for start in range(
                0,
                len(page_embs),
                doc_batch_size,
            ):

                docs = (
                    page_embs[
                        start:
                        start
                        + doc_batch_size
                    ]
                )

                # Different pages can technically
                # have different lengths.
                #
                # score_multi_vector accepts lists.

                q_gpu = [
                    q.squeeze(0)
                    .to(
                        self.device
                    )
                ]

                docs_gpu = [
                    d.to(
                        self.device
                    )
                    for d in docs
                ]

                scores = (
                    self.processor
                    .score_multi_vector(
                        q_gpu,
                        docs_gpu,
                    )
                )

                score_parts.append(
                    scores[
                        0
                    ]
                    .detach()
                    .cpu()
                )

            all_scores = (
                torch.cat(
                    score_parts
                )
            )

            rank = (
                torch.argsort(
                    all_scores,
                    descending=True,
                )
                .numpy()
            )

            rankings.append(
                rank
            )

        return np.asarray(
            rankings
        )


# ============================================================
# SAME VLM READER
# ============================================================

class VLMReader:

    def __init__(
        self,
        model_name,
    ):

        print(
            "\nLoading VLM:",
            model_name,
        )

        self.processor = (
            AutoProcessor
            .from_pretrained(
                model_name
            )
        )

        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                model_name,
                torch_dtype=(
                    MODEL_DTYPE
                ),
                device_map="auto",
            )
        )

        self.model.eval()


    @torch.inference_mode()
    def answer(
        self,
        image,
        question,
    ):

        messages = [
            {
                "role":
                    "user",

                "content": [
                    {
                        "type":
                            "image",

                        "image":
                            image,
                    },

                    {
                        "type":
                            "text",

                        "text":
                            (
                                "Read the document "
                                "image and answer "
                                "the question. "
                                "Return only the "
                                "short answer. "
                                "Do not provide "
                                "an explanation.\n\n"
                                f"Question: "
                                f"{question}"
                            ),
                    },
                ],
            }
        ]

        text = (
            self.processor
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = (
            self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
        )

        # Qwen device-map safe input device
        input_device = next(
            parameter.device
            for parameter
            in self.model.parameters()
            if (
                parameter.device.type
                != "meta"
            )
        )

        inputs = {
            key:
                value.to(
                    input_device
                )
            for key, value
            in inputs.items()
        }

        output = (
            self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )
        )

        generated = (
            output[
                :,
                inputs[
                    "input_ids"
                ].shape[1]:
            ]
        )

        prediction = (
            self.processor
            .batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )

        return (
            prediction.strip()
        )


# ============================================================
# VLM EM/F1
# ============================================================

def evaluate_vlm(
    method_name,
    rankings,
    corpus,
    queries,
    reader,
    max_queries,
):

    n = min(
        max_queries,
        len(queries),
    )

    em_sum = 0.0

    f1_sum = 0.0

    retrieval_hits = 0

    print(
        "\n"
        + "=" * 80
    )

    print(
        "VLM:",
        method_name,
    )

    print(
        "=" * 80
    )

    for i in tqdm(
        range(n)
    ):

        query = (
            queries[i]
        )

        top_idx = int(
            rankings[
                i
            ][0]
        )

        document = (
            corpus[
                top_idx
            ]
        )

        prediction = (
            reader.answer(
                document[
                    "image"
                ],
                query[
                    "question"
                ],
            )
        )

        em = (
            exact_match(
                prediction,
                query[
                    "answers"
                ],
            )
        )

        f1 = (
            token_f1(
                prediction,
                query[
                    "answers"
                ],
            )
        )

        hit = (
            document[
                "page_id"
            ]
            ==
            query[
                "page_id"
            ]
        )

        em_sum += em

        f1_sum += f1

        retrieval_hits += int(
            hit
        )

        if (
            (i + 1)
            % 25
            == 0
        ):

            print(
                f"\n"
                f"{method_name} "
                f"n={i+1} "
                f"R@1="
                f"{retrieval_hits/(i+1):.4f} "
                f"EM="
                f"{em_sum/(i+1):.4f} "
                f"F1="
                f"{f1_sum/(i+1):.4f}"
            )

    return {
        "VLM_R@1":
            retrieval_hits / n,

        "EM":
            em_sum / n,

        "F1":
            f1_sum / n,
    }


# ============================================================
# ORACLE VLM
# ============================================================

def evaluate_oracle(
    corpus,
    queries,
    reader,
    max_queries,
):

    page_to_idx = {
        doc["page_id"]: i
        for i, doc
        in enumerate(corpus)
    }

    n = min(
        max_queries,
        len(queries),
    )

    em_sum = 0.0

    f1_sum = 0.0

    print(
        "\n"
        + "=" * 80
    )

    print(
        "VLM ORACLE GOLD PAGE"
    )

    print(
        "=" * 80
    )

    for i in tqdm(
        range(n)
    ):

        query = queries[i]

        doc = corpus[
            page_to_idx[
                query[
                    "page_id"
                ]
            ]
        ]

        prediction = (
            reader.answer(
                doc[
                    "image"
                ],
                query[
                    "question"
                ],
            )
        )

        em_sum += (
            exact_match(
                prediction,
                query[
                    "answers"
                ],
            )
        )

        f1_sum += (
            token_f1(
                prediction,
                query[
                    "answers"
                ],
            )
        )

    return {
        "EM":
            em_sum / n,

        "F1":
            f1_sum / n,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--train_limit",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--val_limit",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--dense_steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--colpali_steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--dense_batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--colpali_batch",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--colpali_grad_accum",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--em_queries",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--skip_dense_train",
        action="store_true",
    )

    parser.add_argument(
        "--skip_colpali_train",
        action="store_true",
    )

    parser.add_argument(
        "--skip_vlm",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )


    # ========================================================
    # DATASET
    # ========================================================

    train_ds, val_ds = (
        load_docvqa(
            args.train_limit,
            args.val_limit,
        )
    )

    val_corpus, val_queries = (
        build_corpus(
            val_ds
        )
    )

    results = {}

    rankings = {}


    # ========================================================
    # 1. OCR + BM25
    # ========================================================

    bm25_rankings = (
        run_bm25(
            val_corpus,
            val_queries,
        )
    )

    rankings[
        "OCR + BM25"
    ] = bm25_rankings

    results[
        "OCR + BM25"
    ] = retrieval_metrics(
        bm25_rankings,
        val_queries,
        val_corpus,
    )


    # ========================================================
    # 2. OCR + Dense
    # ========================================================

    if (
        args.skip_dense_train
        and
        os.path.isdir(
            DENSE_SAVE_DIR
        )
    ):

        dense_model_source = (
            DENSE_SAVE_DIR
        )

    else:

        dense_model_source = (
            DENSE_MODEL_NAME
        )


    dense = DenseRetriever(
        dense_model_source
    )


    if not (
        args.skip_dense_train
    ):

        dense.train(
            train_ds,
            batch_size=(
                args.dense_batch
            ),
            max_steps=(
                args.dense_steps
            ),
        )

        dense.save(
            DENSE_SAVE_DIR
        )


    dense_rankings = (
        dense.retrieve(
            val_corpus,
            val_queries,
        )
    )

    rankings[
        "OCR + Dense FT"
    ] = dense_rankings

    results[
        "OCR + Dense FT"
    ] = retrieval_metrics(
        dense_rankings,
        val_queries,
        val_corpus,
    )


    np.save(
        os.path.join(
            OUTPUT_DIR,
            "dense_rankings.npy",
        ),
        dense_rankings,
    )


    del dense

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # ========================================================
    # 3. ColPali PRETRAINED
    # ========================================================

    colpali = (
        ColPaliRetriever(
            COLPALI_MODEL_NAME
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "COLPALI PRETRAINED EVAL"
    )

    print(
        "=" * 80
    )

    colpali_pre_rankings = (
        colpali.retrieve(
            val_corpus,
            val_queries,
        )
    )

    rankings[
        "ColPali pretrained"
    ] = colpali_pre_rankings

    results[
        "ColPali pretrained"
    ] = retrieval_metrics(
        colpali_pre_rankings,
        val_queries,
        val_corpus,
    )


    # ========================================================
    # 4. ColPali FT
    # ========================================================

    if not (
        args.skip_colpali_train
    ):

        colpali.train(
            train_ds,
            batch_size=(
                args.colpali_batch
            ),
            max_steps=(
                args.colpali_steps
            ),
            grad_accum=(
                args.colpali_grad_accum
            ),
        )

        colpali.save(
            COLPALI_SAVE_DIR
        )


        print(
            "\nColPali FT retrieval..."
        )

        colpali_ft_rankings = (
            colpali.retrieve(
                val_corpus,
                val_queries,
            )
        )

        rankings[
            "ColPali FT"
        ] = colpali_ft_rankings

        results[
            "ColPali FT"
        ] = retrieval_metrics(
            colpali_ft_rankings,
            val_queries,
            val_corpus,
        )


        np.save(
            os.path.join(
                OUTPUT_DIR,
                "colpali_ft_rankings.npy",
            ),
            colpali_ft_rankings,
        )


    del colpali

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # ========================================================
    # VLM EM / F1
    # ========================================================

    if not args.skip_vlm:

        reader = (
            VLMReader(
                VLM_MODEL_NAME
            )
        )

        # Evaluate all retrieval methods
        # with the SAME page-image VLM reader.

        for (
            method_name,
            method_rankings
        ) in rankings.items():

            vlm_result = (
                evaluate_vlm(
                    method_name,
                    method_rankings,
                    val_corpus,
                    val_queries,
                    reader,
                    args.em_queries,
                )
            )

            results[
                method_name
            ].update(
                vlm_result
            )


        oracle = (
            evaluate_oracle(
                val_corpus,
                val_queries,
                reader,
                args.em_queries,
            )
        )

        results[
            "Oracle Gold Page"
        ] = {
            "Recall@1":
                1.0,

            "Recall@5":
                1.0,

            "MRR":
                1.0,

            "VLM_R@1":
                1.0,

            "EM":
                oracle[
                    "EM"
                ],

            "F1":
                oracle[
                    "F1"
                ],
        }


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n\n"
        + "=" * 110
    )

    print(
        "FINAL DOCVQA ABLATION"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Method':28s}"
        f"{'R@1':>12s}"
        f"{'R@5':>12s}"
        f"{'MRR':>12s}"
        f"{'VLM EM':>12s}"
        f"{'VLM F1':>12s}"
    )

    print(
        "-" * 110
    )

    for (
        method,
        metric
    ) in results.items():

        print(
            f"{method:28s}"
            f"{metric.get('Recall@1', 0):12.4f}"
            f"{metric.get('Recall@5', 0):12.4f}"
            f"{metric.get('MRR', 0):12.4f}"
            f"{metric.get('EM', 0):12.4f}"
            f"{metric.get('F1', 0):12.4f}"
        )


    # ========================================================
    # JSON
    # ========================================================

    with open(
        os.path.join(
            OUTPUT_DIR,
            "results.json",
        ),
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )


    print(
        "\nSaved results:",
        os.path.join(
            OUTPUT_DIR,
            "results.json",
        ),
    )


if __name__ == "__main__":
    main()
