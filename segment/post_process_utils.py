#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------- 基本 I/O / 工具 ----------------

def read_image_wh_from_array(img: np.ndarray) -> Tuple[int, int]:
    """回傳 (W, H)"""
    assert img is not None and img.ndim == 3
    h, w = img.shape[:2]
    return w, h

def _to_float_list(tokens: List[str]) -> List[float]:
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except Exception:
            pass
    return out

# ---------------- YOLO-Seg 行 解析/格式化 ----------------

def parse_yolo_seg_line(line: str):
    """
    支援：
      1) 單段：class x1 y1 x2 y2 ... [conf?]
      2) 多段：class x1 y1 ... | x1 y1 ... | ... [conf?]
    回傳：(class_id:int 或 None, [poly1, poly2, ...])；poly=[x1,y1,x2,y2,...] (0~1)
    非 segmentation（如 bbox）→ (None, [])
    """
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None, []

    if "|" in raw:
        parts = [p.strip() for p in raw.split("|")]
        t0 = parts[0].split()
        if len(t0) < 3:
            return None, []
        try:
            cid = int(float(t0[0]))
        except Exception:
            return None, []
        segs = []
        v0 = _to_float_list(t0[1:])
        if len(v0) >= 7 and (len(v0) % 2 == 1):
            v0 = v0[:-1]  # 丟 conf
        if len(v0) >= 6 and len(v0) % 2 == 0:
            segs.append(v0)
        for p in parts[1:]:
            vv = _to_float_list(p.split())
            if len(vv) >= 7 and (len(vv) % 2 == 1):
                vv = vv[:-1]
            if len(vv) >= 6 and len(vv) % 2 == 0:
                segs.append(vv)
        return cid, segs

    toks = raw.split()
    if len(toks) < 3:
        return None, []
    try:
        cid = int(float(toks[0]))
    except Exception:
        return None, []
    vals = _to_float_list(toks[1:])
    if len(vals) in (4, 5):  # bbox → 略過
        return None, []
    if len(vals) >= 7 and (len(vals) % 2 == 1):
        vals = vals[:-1]
    if len(vals) >= 6 and len(vals) % 2 == 0:
        return cid, [vals]
    return None, []

def format_yolo_seg_line(cid: int, polys: List[List[float]], precision: int = 6) -> str:
    def fmt(vals: List[float]) -> str:
        return " ".join(f"{v:.{precision}f}" for v in vals)
    if not polys:
        return ""
    s = f"{cid} {fmt(polys[0])}"
    for poly in polys[1:]:
        s += " | " + fmt(poly)
    return s

# ---------------- Step 1：裁後 0~1 → 原圖 0~1（單張版） ----------------

def map_coords_to_original(coords: List[float],
                           Wc: int, Hc: int,
                           add_l: int, add_r: int, add_t: int, add_b: int) -> List[float]:
    """
    x' = (x*Wc + add_l) / (Wc + add_l + add_r)
    y' = (y*Hc + add_t) / (Hc + add_t + add_b)
    """
    pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
    Wo = Wc + add_l + add_r
    Ho = Hc + add_t + add_b
    xn = (pts[:, 0] * Wc + add_l) / float(Wo) if (add_l or add_r) else pts[:, 0]
    yn = (pts[:, 1] * Hc + add_t) / float(Ho) if (add_t or add_b) else pts[:, 1]
    out = np.stack([np.clip(xn, 0.0, 1.0), np.clip(yn, 0.0, 1.0)], 1).reshape(-1).tolist()
    return out

def step1_map_to_original_single(lines: List[str],
                                 Wc: int, Hc: int,
                                 add_left: int, add_right: int, add_top: int, add_bottom: int
                                 ) -> List[str]:
    """對單張影像的 seg 行做映射，仍維持 0~1。"""
    mapped_lines: List[str] = []
    for line in lines:
        cid, segs = parse_yolo_seg_line(line)
        if cid is None or not segs:
            continue
        mapped = []
        for poly in segs:
            if len(poly) >= 6 and len(poly) % 2 == 0:
                mapped.append(map_coords_to_original(poly, Wc, Hc,
                                                     add_left, add_right, add_top, add_bottom))
        if mapped:
            mapped_lines.append(format_yolo_seg_line(cid, mapped, precision=6))
    return mapped_lines

# ---------------- Step 2：y 鏡射（0~1；單張版） ----------------

def flip_y_normalized_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        cid, segs = parse_yolo_seg_line(line)
        if cid is None or not segs:
            continue
        newpolys = []
        for poly in segs:
            arr = np.array(poly, dtype=np.float64).reshape(-1, 2)
            arr[:, 1] = 1.0 - arr[:, 1]
            arr = np.clip(arr, 0.0, 1.0)
            newpolys.append(arr.reshape(-1).tolist())
        out.append(format_yolo_seg_line(cid, newpolys, precision=6))
    return out

# ---------------- Step 3：輸出單行（每物件頂點 + 內部像素） ----------------

def scale_norm_to_int(coords: List[float], w: int, h: int, rounding: str) -> List[int]:
    arr = np.array(coords, dtype=np.float64).reshape(-1, 2)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0) * (w - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0) * (h - 1)
    if rounding == "round":
        arr = np.rint(arr)  # ties-to-even
    elif rounding == "floor":
        arr = np.floor(arr)
    else:
        arr = np.ceil(arr)
    arr = np.clip(arr, [0, 0], [w - 1, h - 1]).astype(np.int64)
    return arr.reshape(-1).tolist()

def _bl_ints_to_tl_polygon(int_pairs: List[int], h: int) -> np.ndarray:
    """底左 → 頂左（OpenCV 填充用）"""
    pts = np.array(int_pairs, dtype=np.int64).reshape(-1, 2)
    pts[:, 1] = (h - 1) - pts[:, 1]
    return pts.reshape(-1, 1, 2).astype(np.int32)

def _iter_mask_points(polys_tl: List[np.ndarray], w: int, h: int, stride: int = 1):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts=polys_tl, color=1)
    ys, xs = np.nonzero(mask)
    if stride > 1:
        keep = (ys % stride == 0) & (xs % stride == 0)
        ys, xs = ys[keep], xs[keep]
    yb = (h - 1) - ys
    xb = xs
    order = np.lexsort((xb, yb))  # 穩定輸出
    for i in order:
        yield int(xb[i]), int(yb[i])

def emit_one_line_for_image(lines: List[str], w: int, h: int,
                            rounding: str = "round",
                            include_interior: bool = True,
                            interior_stride: int = 1,
                            dedup_points: bool = False) -> str:
    """
    產出單行：obj_id x y obj_id x y ...
    - 頂點後接內部像素（底左整數座標）
    """
    tokens: List[str] = []
    obj_id = 0
    for line in lines:
        cid, segs = parse_yolo_seg_line(line)
        if cid is None or not segs:
            continue
        # 頂點（0~1 → 像素整數；底左）
        verts_norm = []
        for poly in segs:
            if len(poly) >= 6 and len(poly) % 2 == 0:
                verts_norm.extend(poly)
        if not verts_norm:
            obj_id += 1
            continue
        verts_int = scale_norm_to_int(verts_norm, w, h, rounding)

        seen = set() if dedup_points else None
        for i in range(0, len(verts_int), 2):
            x, y = int(verts_int[i]), int(verts_int[i + 1])
            if seen is not None and (x, y) in seen:
                continue
            if seen is not None:
                seen.add((x, y))
            tokens += [str(obj_id), str(x), str(y)]

        if include_interior:
            polys_tl = []
            for poly in segs:
                ints = scale_norm_to_int(poly, w, h, rounding)
                polys_tl.append(_bl_ints_to_tl_polygon(ints, h))
            for x, y in _iter_mask_points(polys_tl, w, h, stride=interior_stride):
                if seen is not None and (x, y) in seen:
                    continue
                if seen is not None:
                    seen.add((x, y))
                tokens += [str(obj_id), str(x), str(y)]
        obj_id += 1
    return " ".join(tokens)

# ---------------- 視覺化：把 Step1 的結果畫回原圖 ----------------

def draw_step1_on_original(img_bgr: np.ndarray, lines_step1: List[str],
                           edge_thickness: int = 2,
                           edge_color: Tuple[int, int, int] = (0, 255, 0),
                           fill_alpha: float = 0.25) -> np.ndarray:
    """
    lines_step1：已映射回原圖 0~1（頂左原點）的 seg 行
    以頂左原點畫到原圖（OpenCV），可半透明填色。
    """
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()
    if fill_alpha > 0:
        overlay = vis.copy()
    for line in lines_step1:
        cid, segs = parse_yolo_seg_line(line)
        if cid is None or not segs:
            continue
        for poly in segs:
            arr = np.array(poly, dtype=np.float64).reshape(-1, 2)
            pts = np.stack([arr[:, 0] * (w - 1), arr[:, 1] * (h - 1)], 1).round().astype(np.int32)
            pts = pts.reshape(-1, 1, 2)
            if fill_alpha > 0:
                cv2.fillPoly(overlay, [pts], color=edge_color)
            cv2.polylines(vis, [pts], isClosed=True, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)
    if fill_alpha > 0:
        vis = cv2.addWeighted(overlay, fill_alpha, vis, 1 - fill_alpha, 0)
    return vis

