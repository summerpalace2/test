"""模拟示例数据（结构与 UCI Cleveland 心脏病数据集一致，数值为随机生成，仅用于演示/测试）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .medical import HEART_COLUMNS

SAMPLE_NAME = "heart_cleveland_模拟示例"


def make_heart_sample(n: int = 303, seed: int = 42, missing: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = np.array([0.54, 0.18, 0.12, 0.115, 0.045])
    num = rng.choice(5, size=n, p=probs / probs.sum())
    sick = (num > 0).astype(int)
    severity = np.where(sick == 1, num, 0)

    age = np.clip(rng.normal(52 + 5 * sick, 9), 29, 77).round()
    sex = (rng.random(n) < np.where(sick == 1, 0.82, 0.55)).astype(int)
    cp = np.array([rng.choice([1, 2, 3, 4], p=([0.12, 0.25, 0.38, 0.25] if s == 0 else [0.05, 0.10, 0.20, 0.65]))
                   for s in sick])
    trestbps = np.clip(rng.normal(129 + 5 * sick, 16), 94, 200).round()
    chol = np.clip(rng.normal(243 + 11 * sick, 50), 126, 564).round()
    fbs = (rng.random(n) < np.where(sick == 1, 0.16, 0.14)).astype(int)
    restecg = np.array([rng.choice([0, 1, 2], p=([0.58, 0.01, 0.41] if s == 0 else [0.40, 0.02, 0.58]))
                        for s in sick])
    thalach = np.clip(rng.normal(158 - 19 * sick - 2 * severity, 19), 71, 202).round()
    exang = (rng.random(n) < np.where(sick == 1, 0.55, 0.14)).astype(int)
    oldpeak = np.clip(rng.gamma(1.5, np.where(sick == 1, 0.95, 0.4)), 0, 6.2).round(1)
    slope = np.array([rng.choice([1, 2, 3], p=([0.65, 0.30, 0.05] if s == 0 else [0.25, 0.65, 0.10]))
                      for s in sick])
    ca = np.array([rng.choice([0, 1, 2, 3], p=([0.78, 0.15, 0.05, 0.02] if s == 0 else [0.30, 0.32, 0.22, 0.16]))
                   for s in sick]).astype(float)
    thal = np.array([rng.choice([3, 6, 7], p=([0.78, 0.07, 0.15] if s == 0 else [0.30, 0.06, 0.64]))
                     for s in sick]).astype(float)

    df = pd.DataFrame(dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg,
                           thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal, num=num))
    df = df[HEART_COLUMNS]
    if missing:  # 与真实数据类似：ca 少量、thal 极少量缺失
        df.loc[rng.choice(n, 4, replace=False), "ca"] = np.nan
        df.loc[rng.choice(n, 2, replace=False), "thal"] = np.nan
    return df


def write_samples(out_dir: str | Path) -> dict[str, Path]:
    """生成 csv / 无表头 txt / xlsx 三种格式及一个图像示例目录，供测试与演示。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = make_heart_sample()
    paths = {}
    p = out / "processed.cleveland.csv"
    df.to_csv(p, index=False)
    paths["csv"] = p
    p = out / "processed.cleveland.data.txt"  # UCI 原始格式：无表头，缺失为 ?
    df.to_csv(p, index=False, header=False, na_rep="?")
    paths["txt"] = p
    p = out / "heart_sample.xlsx"
    df.to_excel(p, index=False)
    paths["xlsx"] = p
    paths["images"] = write_sample_images(out / "images")
    return paths


def write_sample_images(out_dir: str | Path, per_class: int = 8, size: int = 96, seed: int = 0) -> Path:
    """生成两类简单灰度“影像”：normal（平滑噪声）与 lesion（带亮斑），用于演示图像模块。"""
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    for label in ("normal", "lesion"):
        (out / label).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            base = rng.normal(110, 25, (size, size))
            img = Image.fromarray(np.clip(base, 0, 255).astype("uint8"), "L").filter(ImageFilter.GaussianBlur(3))
            arr = np.asarray(img).astype(float)
            if label == "lesion":
                cx, cy = rng.integers(25, size - 25, 2)
                yy, xx = np.mgrid[:size, :size]
                arr += 90 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rng.uniform(5, 10) ** 2)))
            Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "L").save(out / label / f"{label}_{i + 1:02d}.png")
    return out


if __name__ == "__main__":  # python -m medminer.core.sample_data  → 重新生成 resources/sample
    out = Path(__file__).resolve().parents[2] / "resources" / "sample"
    for k, v in write_samples(out).items():
        print(k, v)
