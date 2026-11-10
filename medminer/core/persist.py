"""模型保存/载入（joblib）与数据泄漏检测。

安全提示：joblib/pickle 文件在载入时可执行任意代码，只应载入自己生成或完全信任的模型文件。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .. import __version__
from .schema import BINARY, CATEGORICAL, NUMERIC

MAGIC = "medminer-model-v1"


@dataclass
class ModelPackage:
    name: str
    pipeline: object
    task: str
    target: str
    features: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    classes: list[str]
    threshold: float = 0.5
    metrics: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)  # 数值列 -> (min, median, max)
    categories: dict = field(default_factory=dict)  # 分类列 -> 取值列表
    n_train: int = 0
    created: str = ""
    version: str = __version__
    spec_key: str = ""  # 用于选择 SHAP 解释器


def describe_features(X: pd.DataFrame, num_cols: list[str], cat_cols: list[str]):
    ranges = {c: (float(X[c].min()), float(X[c].median()), float(X[c].max())) for c in num_cols if X[c].notna().any()}
    cats = {}
    for c in cat_cols:
        vals = sorted(X[c].dropna().unique(), key=lambda v: (str(type(v)), v))
        cats[c] = [v.item() if hasattr(v, "item") else v for v in vals]
    return ranges, cats


def make_package(name, pipeline, task, target, X: pd.DataFrame, num_cols, cat_cols, classes, threshold, metrics,
                 spec_key="") -> ModelPackage:
    ranges, cats = describe_features(X, num_cols, cat_cols)
    return ModelPackage(name, pipeline, task, target, list(X.columns), list(num_cols), list(cat_cols), list(classes),
                        float(threshold), {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))},
                        ranges, cats, int(len(X)), dt.datetime.now().isoformat(timespec="seconds"), __version__, spec_key)


def save_package(pkg: ModelPackage, path: str | Path) -> None:
    joblib.dump({"magic": MAGIC, "package": pkg}, path, compress=3)


def load_package(path: str | Path) -> ModelPackage:
    obj = joblib.load(path)
    if not isinstance(obj, dict) or obj.get("magic") != MAGIC or not isinstance(obj.get("package"), ModelPackage):
        raise ValueError("这不是 MedMiner 生成的模型文件")
    return obj["package"]


def predict_frame(pkg: ModelPackage, df: pd.DataFrame) -> pd.DataFrame:
    """批量预测：返回 [预测, 概率…]。缺少的特征列会报错；多余的列忽略。"""
    missing = [c for c in pkg.features if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少模型需要的列：{missing}")
    X = df[pkg.features]
    if pkg.task == "regression":
        return pd.DataFrame({"预测值": pkg.pipeline.predict(X)})
    proba = pkg.pipeline.predict_proba(X)
    out = {}
    if len(pkg.classes) == 2:
        p = proba[:, 1]
        out["阳性概率"] = p
        out["预测类别"] = np.where(p >= pkg.threshold, pkg.classes[1], pkg.classes[0])
    else:
        for i, c in enumerate(pkg.classes):
            out[f"P({c})"] = proba[:, i]
        out["预测类别"] = [pkg.classes[i] for i in proba.argmax(1)]
    return pd.DataFrame(out)


# ---------------- 数据泄漏检测 ----------------
def leakage_suspects(X: pd.DataFrame, y: np.ndarray, types: dict[str, str], task: str) -> list[tuple[str, str]]:
    """找出几乎能“直接推出”目标的特征（常见原因：目标由某列二值化得来，却把原列留在特征里）。
    返回 [(列名, 原因)]。"""
    out = []
    yy = pd.Series(y)
    for c in X.columns:
        s = X[c]
        ok = s.notna()
        if ok.sum() < 20:
            continue
        if types.get(c) in (CATEGORICAL, BINARY) and task == "classification":
            if s[ok].nunique() < 2:
                continue
            ct = pd.crosstab(s[ok], yy[ok])
            few = s[ok].nunique() <= max(12, int(ok.sum() * 0.05))  # 唯一值太多时纯度天然偏高，只在取值较少时判定
            fwd = ct.max(axis=1).sum() / ok.sum()  # 特征 → 目标：特征的每个取值几乎只对应一个目标类
            rev = ct.max(axis=0).sum() / ok.sum()  # 目标 → 特征：每个目标类几乎只对应一个特征取值
            if fwd >= 0.99 and few:
                out.append((c, f"该列的每个取值几乎只对应一个目标类别（纯度 {fwd:.1%}），疑似目标本身或其直接编码"))
            elif rev >= 0.99 and few and ct.shape[0] >= 2 and ct.shape[1] <= 12:
                out.append((c, f"目标的每个类别几乎只对应该列的一个取值（纯度 {rev:.1%}），疑似由目标派生（如目标二值化后的列）"))
        elif types.get(c) == NUMERIC:
            try:
                if task == "classification" and len(np.unique(yy[ok])) == 2:
                    auc = roc_auc_score(yy[ok], s[ok])
                    if max(auc, 1 - auc) >= 0.995:
                        out.append((c, f"单独使用该列的 AUC≈{max(auc, 1 - auc):.3f}，几乎完美区分目标，疑似泄漏"))
                elif task == "regression":
                    r = abs(pd.Series(s[ok]).corr(yy[ok], method="spearman"))
                    if r >= 0.98:
                        out.append((c, f"与目标的 Spearman 相关高达 {r:.3f}，疑似由目标直接推出"))
            except ValueError:
                continue
    return out
