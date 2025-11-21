#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

from model_predict_func import SegPredictor
from post_process_utils import (
    read_image_wh_from_array,
    step1_map_to_original_single,
    flip_y_normalized_lines,
    emit_one_line_for_image,
    draw_step1_on_original,
)

# ===================== 使用者在這裡填參數 =====================

# 1) 影像路徑（裁切前原圖）
IMAGE_PATH = "/data/yolov9/data/images/img_1141015101033.png"

# 2) 裁切像素（等於當初被裁掉的邊界量）
ADD_LEFT = 0  # 263
ADD_RIGHT = 30  # 241
ADD_TOP = 0
ADD_BOTTOM = 0

# 3) 模型設定
SHIP_WEIGHTS_PATH = "/data/yolov9/segment/ship.pt"
SEA_WEIGHTS_PATH = "/data/yolov9/segment/sea.pt"
IMG_SIZE = (1080, 1080)  # 視你的模型設定
CONF_THRES = 0.05
IOU_THRES = 0.05
MAX_DET = 1000
SHIP_PREDICTOR = SegPredictor(SHIP_WEIGHTS_PATH, imgsz=IMG_SIZE)
SEA_PREDICTOR = SegPredictor(SEA_WEIGHTS_PATH, imgsz=IMG_SIZE)
# 4) 後處理 Step3 選項
ROUNDING = "round"  # "round" | "floor" | "ceil"
INTERIOR_STRIDE = 1  # 內部像素下採樣（1=全點；4/8 可大幅降檔）
DEDUP_POINTS = True  # 去除同一物件頂點/內部的重複點
FORCE_LINE = True  # 即使沒有物件仍輸出一行
EMPTY_MODE = "blank"  # "blank" 或 "minus1"

# 5) 輸出路徑
LABELS_OUT_DIR = "/data/yolov9/segment/label_out"  # Step3 之後的單行 txt
STEP1_VIS_DIR = "/data/yolov9/segment/img_out"  # 畫在原圖上的可視化（Step1 後）
BACKUP_LABELS_OUT_DIR = None
BACKUP_VIS_DIR = None

STEP1_EDGE_THK = 2  # 可視化線寬
STEP1_EDGE_COLOR = (0, 0, 255)  # BGR
STEP1_FILL_ALPHA = 0.25  # 0~1 半透明填色

# ============================================================


def safe_imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def crop_in_memory(
    img: np.ndarray, add_left: int, add_right: int, add_top: int, add_bottom: int
) -> np.ndarray:
    """只在記憶體做裁切，不落地。"""
    h, w = img.shape[:2]
    x0 = max(0, int(add_left))
    x1 = max(0, w - int(add_right))
    y0 = max(0, int(add_top))
    y1 = max(0, h - int(add_bottom))
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"無效裁切範圍：({x0},{y0})-({x1},{y1}) for image {w}x{h}")
    return img[y0:y1, x0:x1, :].copy()


def save_text_line(out_dir: Path, stem: str, content: str, mode: str, force_line: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "ship":
        prefix = "txt_"
    else:
        prefix = "txt_current_"
    # 若 content == "" 且 force_line=True，仍寫入換行
    txt = (
        (content + "\n")
        if (force_line and content == "")
        else (content + ("\n" if content and not content.endswith("\n") else ""))
    )
    (out_dir / f"{prefix}{stem}.txt").write_text(txt, encoding="utf-8")


def save_image(
    out_dir: Path, stem: str, img_bgr: np.ndarray, mode: str, suffix: str = ".jpg"
):
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "ship":
        prefix = "img_"
    else:
        prefix = "img_current_"
    cv2.imencode(suffix, img_bgr)[1].tofile(str(out_dir / f"{prefix}{stem}{suffix}"))


def run(mode: str = "ship"):
    img_path = Path(IMAGE_PATH)
    stem = img_path.stem.split("_")[1]

    # 讀原圖（裁切前）
    img0 = safe_imread(img_path)
    if img0 is None:
        raise FileNotFoundError(f"Error read image：{img_path}")
    Ho, Wo = img0.shape[:2]

    # in-memory 裁切（不落地）
    img_crop = crop_in_memory(img0, ADD_LEFT, ADD_RIGHT, ADD_TOP, ADD_BOTTOM)
    Wc, Hc = read_image_wh_from_array(img_crop)

    # 模型推論（直接吃裁切後影像的 ndarray）
    if mode == "ship":
        predictor = SHIP_PREDICTOR
        img_prefix = "img_"
        txt_prefix = "txt_"
    else:
        predictor = SEA_PREDICTOR
        img_prefix = "img_current_"
        txt_prefix = "txt_current_"

    lines_cropped = predictor.predict_one(
        img_crop, conf_thres=CONF_THRES, iou_thres=IOU_THRES, max_det=MAX_DET
    )
    # lines_cropped 是「裁後影像」上的 YOLO-Seg 單行字串（0~1）

    # Step 1：映射回「原圖 0~1」
    lines_step1 = step1_map_to_original_single(
        lines_cropped,
        Wc=Wc,
        Hc=Hc,
        add_left=ADD_LEFT,
        add_right=ADD_RIGHT,
        add_top=ADD_TOP,
        add_bottom=ADD_BOTTOM,
    )

    # 把 Step1 結果畫回「原圖」並輸出
    vis = draw_step1_on_original(
        img0,
        lines_step1,
        edge_thickness=STEP1_EDGE_THK,
        edge_color=STEP1_EDGE_COLOR,
        fill_alpha=STEP1_FILL_ALPHA,
    )
    save_image(Path(STEP1_VIS_DIR), stem, vis, mode, suffix=".png")
    if BACKUP_VIS_DIR is not None:
        save_image(Path(BACKUP_VIS_DIR), stem, vis, mode, suffix=".png")

    # Step 2：y 鏡射（仍 0~1；轉成底左原點的正規化）
    lines_step2 = flip_y_normalized_lines(lines_step1)

    # Step 3：輸出單行（每物件頂點 + 內部像素；像素整數、底左原點）
    w_orig, h_orig = (
        Wo,
        Ho,
    )  # 注意 read_image_wh 的回傳順序在 utils 版是 (W,H)，這裡我們已經有 Ho,Wo
    final_line = emit_one_line_for_image(
        lines_step2,
        w=w_orig,
        h=h_orig,
        rounding=ROUNDING,
        include_interior=True,
        interior_stride=INTERIOR_STRIDE,
        dedup_points=DEDUP_POINTS,
    )

    # 存最終 txt
    save_text_line(
        Path(LABELS_OUT_DIR),
        stem,
        final_line,
        mode,
        force_line=(
            FORCE_LINE
            if EMPTY_MODE == "blank"
            else True if EMPTY_MODE == "minus1" else False
        ),
    )
    if BACKUP_LABELS_OUT_DIR is not None:
        save_text_line(
            Path(BACKUP_LABELS_OUT_DIR),
            stem,
            final_line,
            mode,
            force_line=(
                FORCE_LINE
                if EMPTY_MODE == "blank"
                else True if EMPTY_MODE == "minus1" else False
            ),
        )
    # 如果你想在「無物件」時一定輸出 -1，可這樣處理：
    if final_line.strip() == "" and FORCE_LINE and EMPTY_MODE == "minus1":
        save_text_line(Path(LABELS_OUT_DIR), stem, "-1", mode, force_line=False)

    print(
        f"finish：img_{stem}\n"
        f" - Step1 picture：{Path(STEP1_VIS_DIR) / (img_prefix + stem + '.png')}\n"
        f" - Label：{Path(LABELS_OUT_DIR) / (txt_prefix + stem + '.txt')}"
    )


if __name__ == "__main__":
    run('ship')
