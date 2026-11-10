"""图像数据：扫描文件夹（子文件夹名=标签）、读取元信息、提取可用于建模的图像特征。

仅依赖 Pillow + numpy。DICOM 等医学专用格式需先转为 PNG/JPG/TIFF（或自行安装 pydicom 扩展）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .io import IMAGE_EXTS

FEATURE_NAMES = (["gray_mean", "gray_std", "gray_min", "gray_max", "gray_median", "gray_skew", "gray_kurt",
                  "entropy", "contrast_rms", "edge_density", "edge_mean", "sharpness_lapvar"]
                 + [f"hist_bin{i + 1}" for i in range(8)]
                 + ["r_mean", "g_mean", "b_mean", "center_mean", "border_mean", "center_border_diff"])


def scan_image_folder(folder: str | Path, max_images: int = 5000) -> pd.DataFrame:
    """递归扫描。若图片位于一级子文件夹中，则子文件夹名作为 label；否则 label 为空。"""
    root = Path(folder)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        rel = p.relative_to(root)
        label = rel.parts[0] if len(rel.parts) > 1 else ""
        try:
            with Image.open(p) as im:
                w, h = im.size
                mode = im.mode
        except Exception:
            continue  # 损坏文件跳过
        rows.append({"file": str(rel), "path": str(p), "label": label, "width": w, "height": h, "mode": mode,
                     "size_kb": round(p.stat().st_size / 1024, 1)})
        if len(rows) >= max_images:
            break
    return pd.DataFrame(rows, columns=["file", "path", "label", "width", "height", "mode", "size_kb"])


def _entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def image_features(path: str | Path, size: int = 128) -> dict[str, float]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        rgb = im.convert("RGB").resize((size, size))
    arr = np.asarray(rgb, dtype=float)
    gray = np.asarray(rgb.convert("L"), dtype=float)
    g = gray.ravel()
    m, s = g.mean(), g.std()
    z = (g - m) / s if s > 0 else np.zeros_like(g)
    gx = np.abs(np.diff(gray, axis=1))[:-1, :]
    gy = np.abs(np.diff(gray, axis=0))[:, :-1]
    mag = np.hypot(gx, gy)
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    hist, _ = np.histogram(g, bins=8, range=(0, 255))
    q = size // 4
    center = gray[q:-q, q:-q]
    border = np.concatenate([gray[:q].ravel(), gray[-q:].ravel(), gray[:, :q].ravel(), gray[:, -q:].ravel()])
    feats = {"gray_mean": m, "gray_std": s, "gray_min": g.min(), "gray_max": g.max(), "gray_median": float(np.median(g)),
             "gray_skew": float((z ** 3).mean()), "gray_kurt": float((z ** 4).mean() - 3), "entropy": _entropy(gray),
             "contrast_rms": s, "edge_density": float((mag > 20).mean()), "edge_mean": float(mag.mean()),
             "sharpness_lapvar": float(lap.var())}
    feats.update({f"hist_bin{i + 1}": float(v / g.size) for i, v in enumerate(hist)})
    feats.update(r_mean=float(arr[..., 0].mean()), g_mean=float(arr[..., 1].mean()), b_mean=float(arr[..., 2].mean()),
                 center_mean=float(center.mean()), border_mean=float(border.mean()),
                 center_border_diff=float(center.mean() - border.mean()))
    return {k: float(v) for k, v in feats.items()}


def extract_features(info: pd.DataFrame, size: int = 128, progress=None) -> pd.DataFrame:
    """对 scan_image_folder 的结果逐张提取特征，返回 [file, label, 尺寸…, 特征…] 的表格，可直接作为数据集分析。"""
    rows = []
    for i, r in enumerate(info.itertuples(index=False)):
        try:
            f = image_features(r.path, size)
        except Exception:
            f = {k: np.nan for k in FEATURE_NAMES}
        rows.append({"file": r.file, "label": r.label or np.nan, "width": r.width, "height": r.height, **f})
        if progress:
            progress(i + 1, len(info))
    out = pd.DataFrame(rows)
    if out["label"].isna().all():
        out = out.drop(columns="label")
    return out
