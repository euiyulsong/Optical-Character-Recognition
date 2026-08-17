# sam3_train.py
#
# 네 기존 download_det.py 결과를 그대로 사용:
#
# data/
# ├── images/
# │   ├── 000000.jpg
# │   ├── 000001.jpg
# │   └── ...
# └── annotations.json
#
# annotations.json:
# [
#   {
#       "image": "data/images/000000.jpg",
#       "boxes": [...],
#       "texts": [...]
#   },
#   ...
# ]
#
#
# 준비:
#
# git clone https://github.com/facebookresearch/sam3
# cd sam3
# pip install -e ".[train]"
#
# 다시 네 프로젝트 폴더로 와서:
#
# python3 sam3_train.py
#
#
# 이 스크립트는:
#
# 1. 기존 annotations.json 읽기
# 2. train/val split
# 3. bbox를 rectangular segmentation polygon으로 변환
# 4. SAM3용 COCO JSON 생성
# 5. SAM3 example config를 복사
# 6. dataset path 수정
# 7. 공식 training script 실행
#
# IMPORTANT:
#
# bounding box로 만든 mask이므로
# "정확한 character/text pixel mask"가 아니라
# "text region rectangular mask" 학습이다.
#
# DBNet text detection과 비교하는 quick experiment 용도.


import os
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

ANNOTATION_FILE = Path(
    "data/annotations.json"
)

IMAGE_ROOT = Path(
    "data/images"
)

OUTPUT_ROOT = Path(
    "data/sam3_text"
)

TRAIN_DIR = (
    OUTPUT_ROOT
    / "train"
)

VAL_DIR = (
    OUTPUT_ROOT
    / "valid"
)

TRAIN_JSON = (
    TRAIN_DIR
    / "_annotations.coco.json"
)

VAL_JSON = (
    VAL_DIR
    / "_annotations.coco.json"
)


# download_det.py:
# 1000 train + 200 val 기준
NUM_TRAIN = 1000
NUM_VAL = 200


# SAM3 repository 위치
SAM3_ROOT = Path(
    os.environ.get(
        "SAM3_ROOT",
        "../sam3",
    )
)


# SAM3 공식 full finetune example config
BASE_CONFIG = (
    SAM3_ROOT
    / "sam3"
    / "train"
    / "configs"
    / "roboflow_v100"
    / "roboflow_v100_full_ft_100_images.yaml"
)


# 우리가 생성할 config
OUTPUT_CONFIG = (
    SAM3_ROOT
    / "sam3"
    / "train"
    / "configs"
    / "text_detection_quick.yaml"
)


# ============================================================
# HELPERS
# ============================================================

def flatten_box(box):
    """
    CaptionedSynthText bbox 형태를 최대한 유연하게 처리.

    가능한 형태:

    [x1, y1, x2, y2]

    또는

    [
        [x1,y1],
        [x2,y1],
        [x2,y2],
        [x1,y2]
    ]
    """

    # ------------------------------------------
    # bbox = [x1,y1,x2,y2]
    # ------------------------------------------

    if (
        isinstance(box, list)
        and len(box) == 4
        and all(
            isinstance(x, (int, float))
            for x in box
        )
    ):

        x1, y1, x2, y2 = box

        return (
            float(x1),
            float(y1),
            float(x2),
            float(y2),
        )

    # ------------------------------------------
    # polygon = [[x,y], ...]
    # ------------------------------------------

    if (
        isinstance(box, list)
        and len(box) >= 4
        and isinstance(box[0], (list, tuple))
    ):

        xs = [
            float(p[0])
            for p in box
        ]

        ys = [
            float(p[1])
            for p in box
        ]

        return (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    raise ValueError(
        f"Unknown box format: {box}"
    )


# ============================================================
# COCO CONVERSION
# ============================================================

def make_coco(
    rows,
    output_dir,
    output_json,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "text",
                "supercategory": "text",
            }
        ],
    }

    annotation_id = 1

    valid_image_count = 0

    for image_id, row in enumerate(
        rows,
        start=1,
    ):

        src_path = Path(
            row["image"]
        )

        if not src_path.exists():

            print(
                "missing:",
                src_path,
            )

            continue

        # --------------------------------------------------
        # copy image
        # --------------------------------------------------

        dst_path = (
            output_dir
            / src_path.name
        )

        if not dst_path.exists():

            shutil.copy2(
                src_path,
                dst_path,
            )

        # --------------------------------------------------
        # image size
        # --------------------------------------------------

        with Image.open(
            src_path
        ) as img:

            width, height = (
                img.size
            )

        coco["images"].append({
            "id": image_id,
            "file_name": src_path.name,
            "width": width,
            "height": height,
        })

        boxes = row.get(
            "boxes",
            [],
        )

        texts = row.get(
            "texts",
            [],
        )

        # --------------------------------------------------
        # each text region
        # --------------------------------------------------

        for box_idx, box in enumerate(
            boxes
        ):

            try:

                x1, y1, x2, y2 = (
                    flatten_box(box)
                )

            except Exception as e:

                print(
                    "skip box:",
                    box,
                    e,
                )

                continue

            # clamp

            x1 = max(
                0.0,
                min(
                    x1,
                    width - 1,
                ),
            )

            y1 = max(
                0.0,
                min(
                    y1,
                    height - 1,
                ),
            )

            x2 = max(
                0.0,
                min(
                    x2,
                    width,
                ),
            )

            y2 = max(
                0.0,
                min(
                    y2,
                    height,
                ),
            )

            box_w = (
                x2 - x1
            )

            box_h = (
                y2 - y1
            )

            if (
                box_w <= 1
                or box_h <= 1
            ):

                continue

            # --------------------------------------------------
            # rectangular segmentation
            #
            # x1,y1 ------- x2,y1
            #   |             |
            #   |             |
            # x1,y2 ------- x2,y2
            # --------------------------------------------------

            segmentation = [[
                x1,
                y1,

                x2,
                y1,

                x2,
                y2,

                x1,
                y2,
            ]]

            text = (
                str(texts[box_idx])
                if box_idx < len(texts)
                else "text"
            )

            coco[
                "annotations"
            ].append({

                "id": (
                    annotation_id
                ),

                "image_id": (
                    image_id
                ),

                "category_id": 1,

                # COCO bbox:
                # x,y,width,height
                "bbox": [
                    x1,
                    y1,
                    box_w,
                    box_h,
                ],

                "segmentation": (
                    segmentation
                ),

                "area": (
                    box_w
                    * box_h
                ),

                "iscrowd": 0,

                # SAM3 concept text prompt
                #
                # 실제 word text를 넣지 않고
                # 모든 영역을 "text" concept으로 묶는다.
                #
                # 그래야:
                #
                # prompt = "text"
                #
                # → 모든 text instances detect/segment
                #
                "noun_phrase": "text",
            })

            annotation_id += 1

        valid_image_count += 1

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            coco,
            f,
            ensure_ascii=False,
        )

    print()
    print(
        "=" * 70
    )

    print(
        "COCO:",
        output_json,
    )

    print(
        "images:",
        len(
            coco["images"]
        ),
    )

    print(
        "annotations:",
        len(
            coco["annotations"]
        ),
    )

    print(
        "=" * 70
    )


# ============================================================
# BUILD DATASET
# ============================================================

def prepare_dataset():

    print()
    print(
        "=" * 70
    )
    print(
        "PREPARE SAM3 DATASET"
    )
    print(
        "=" * 70
    )

    if not ANNOTATION_FILE.exists():

        raise FileNotFoundError(
            ANNOTATION_FILE
        )

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        rows = json.load(f)

    print(
        "total rows:",
        len(rows),
    )

    train_rows = rows[
        :NUM_TRAIN
    ]

    val_rows = rows[
        NUM_TRAIN:
        NUM_TRAIN + NUM_VAL
    ]

    print(
        "train:",
        len(train_rows),
    )

    print(
        "val:",
        len(val_rows),
    )

    make_coco(
        train_rows,
        TRAIN_DIR,
        TRAIN_JSON,
    )

    make_coco(
        val_rows,
        VAL_DIR,
        VAL_JSON,
    )


# ============================================================
# SANITY CHECK
# ============================================================

def check_dataset():

    print()
    print(
        "=" * 70
    )
    print(
        "DATASET CHECK"
    )
    print(
        "=" * 70
    )

    with open(
        TRAIN_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        coco = json.load(f)

    print(
        "train images:",
        len(
            coco["images"]
        ),
    )

    print(
        "train annotations:",
        len(
            coco["annotations"]
        ),
    )

    print()

    print(
        "example:"
    )

    print(
        json.dumps(
            coco["annotations"][0],
            indent=2,
            ensure_ascii=False,
        )
    )


# ============================================================
# CONFIG GENERATION
# ============================================================

def make_training_config():

    print()
    print(
        "=" * 70
    )
    print(
        "CREATE SAM3 CONFIG"
    )
    print(
        "=" * 70
    )

    if not BASE_CONFIG.exists():

        raise FileNotFoundError(
            "\nSAM3 example config not found:\n"
            f"{BASE_CONFIG}\n\n"
            "Set SAM3_ROOT, e.g.\n\n"
            "export SAM3_ROOT=$HOME/sam3\n"
        )

    text = BASE_CONFIG.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------
    # Official Roboflow config는
    # roboflow_vl_100_root 변수 사용.
    #
    # dataset root를 우리가 만든 디렉터리로 변경.
    # --------------------------------------------------

    absolute_root = (
        OUTPUT_ROOT
        .resolve()
    )

    replacements = [
        (
            "roboflow_vl_100_root:",
            f"roboflow_vl_100_root: {absolute_root}",
        ),
    ]

    lines = []

    replaced_root = False

    for line in text.splitlines():

        if (
            line.strip()
            .startswith(
                "roboflow_vl_100_root:"
            )
        ):

            indent = (
                line[
                    :
                    len(line)
                    - len(
                        line.lstrip()
                    )
                ]
            )

            lines.append(
                f"{indent}"
                f"roboflow_vl_100_root: "
                f"{absolute_root}"
            )

            replaced_root = True

        else:

            lines.append(
                line
            )

    text = "\n".join(
        lines
    )

    if not replaced_root:

        print(
            "WARNING:"
        )

        print(
            "Could not automatically find "
            "'roboflow_vl_100_root' in config."
        )

        print(
            "You may need to edit dataset root manually."
        )

    OUTPUT_CONFIG.write_text(
        text,
        encoding="utf-8",
    )

    print(
        "config:",
        OUTPUT_CONFIG,
    )


# ============================================================
# TRAIN SAM3
# ============================================================

def train_sam3():

    print()
    print("=" * 70)
    print("TRAIN SAM3")
    print("=" * 70)

    command = [
        "python3",
        "sam3/train/train.py",

        "-c",
        "configs/text_detection_quick.yaml",

        "--use-cluster",
        "0",

        "--num-gpus",
        "1",
    ]

    print(" ".join(command))

    env = os.environ.copy()

    env["PYTHONPATH"] = (
        str(SAM3_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    subprocess.run(
        command,
        check=True,
        cwd=str(SAM3_ROOT),
        env=env,
    )
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 70
    )

    print(
        "SAM3 TEXT DETECTION "
        "FINETUNING"
    )

    print(
        "=" * 70
    )

    print(
        "annotation:",
        ANNOTATION_FILE,
    )

    print(
        "sam3 root:",
        SAM3_ROOT,
    )

    # ------------------------------------------
    # 1. existing data -> COCO
    # ------------------------------------------

    prepare_dataset()

    # ------------------------------------------
    # 2. inspect
    # ------------------------------------------

    check_dataset()

    # ------------------------------------------
    # 3. create SAM3 config
    # ------------------------------------------

    make_training_config()

    # ------------------------------------------
    # 4. official SAM3 training
    # ------------------------------------------

    train_sam3()
