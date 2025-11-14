#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import os, sys
import torch
import numpy as np

# 你原本的 YOLO 專案相依（示意，保留原結構）
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 視你的專案而定
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_segments, check_img_size)
from utils.augmentations import letterbox
from utils.segment.general import masks2segments, process_mask

class SegPredictor:
    """
    用法：
      p = SegPredictor(weights="xxx.pt", imgsz=(1080,1080))
      lines = p.predict_one(img_bgr)  # 回傳 YOLO-Seg 單行字串 list（在「裁後影像」的 0~1）
    """
    def __init__(self, weights, device=None, imgsz=(1080, 1080), fp16=False):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=False, fp16=fp16)
        self.model.eval()
        self.imgsz = check_img_size(imgsz, s=self.model.stride)

    @torch.no_grad()
    def predict_one(self, img_bgr: np.ndarray,
                    conf_thres=0.25, iou_thres=0.45, max_det=1000) -> list[str]:
        assert img_bgr is not None and img_bgr.ndim == 3, "img_bgr 必須是 HxWx3"
        img0 = img_bgr.copy()

        # letterbox → tensor
        img = letterbox(img_bgr, new_shape=self.imgsz[0])[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if img.ndim == 3:
            img = img[None]

        # forward + NMS
        pred, proto = self.model(img, augment=False, visualize=False)[:2]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None,
                                   agnostic=False, max_det=max_det, nm=32)

        lines: list[str] = []
        for det in pred:
            if len(det) == 0:
                continue
            masks = process_mask(proto[2].squeeze(0), det[:, 6:], det[:, :4], img.shape[2:], upsample=True)
            seg_list = masks2segments(masks)  # List[np.ndarray(N,2)] in input (letterbox) space
            # 映射回「裁後原圖 img0」並正規化到 0~1
            seg_list_norm01 = [scale_segments(img.shape[2:], seg, img0.shape, normalize=True) for seg in seg_list]
            classes = det[:, 5].to('cpu').numpy().astype(int).tolist()
            for cls, seg in zip(classes, seg_list_norm01):
                parts = " ".join(f"{float(x):.6f} {float(y):.6f}" for x, y in seg)
                lines.append(f"{cls} {parts}")
        return lines

