"""可解释性：折外置换重要性、模型自带重要性/系数、SHAP（含单个患者解释）。"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from .evaluate import EvalConfig, make_pipeline
from .models import ModelSpec


def heldout_permutation_importance(X: pd.DataFrame, y: np.ndarray, spec: ModelSpec, params: dict, num_cols, cat_cols,
                                   cfg: EvalConfig, n_repeats: int = 10, n_splits: int = 5) -> pd.DataFrame:
    """在每个测试折上做置换重要性再平均，避免用训练数据评估导致的虚高。
    分类用 AUC（二分类）/准确率，回归用 R²。"""
    if cfg.task == "classification":
        k = max(2, min(n_splits, int(np.min(np.bincount(y)))))
        cv = StratifiedKFold(k, shuffle=True, random_state=cfg.seed)
        scoring = "roc_auc" if len(np.unique(y)) == 2 else "accuracy"
    else:
        cv = KFold(n_splits, shuffle=True, random_state=cfg.seed)
        scoring = "r2"
    imps = []
    pipe = make_pipeline(spec, params, num_cols, cat_cols, cfg)
    for tr, te in cv.split(X, y):
        p = clone(pipe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.fit(X.iloc[tr], y[tr])
            r = permutation_importance(p, X.iloc[te], y[te], scoring=scoring, n_repeats=n_repeats,
                                       random_state=cfg.seed, n_jobs=1)
        imps.append(r.importances_mean)
    arr = np.array(imps)
    out = pd.DataFrame({"变量": X.columns, "重要性(得分下降)": arr.mean(0), "标准差(折间)": arr.std(0, ddof=1) if len(arr) > 1 else 0.0})
    return out.sort_values("重要性(得分下降)", ascending=False).reset_index(drop=True)


def native_importance(pipe: Pipeline) -> pd.DataFrame | None:
    """变换后特征空间的模型自带重要性（树）/ 系数（线性）。"""
    model = pipe.named_steps["model"]
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    if hasattr(model, "feature_importances_"):
        v = np.asarray(model.feature_importances_, float)
        kind = "特征重要性"
    elif hasattr(model, "coef_"):
        c = np.asarray(model.coef_, float)
        v = c[0] if c.ndim == 2 and c.shape[0] == 1 else (np.abs(c).mean(0) if c.ndim == 2 else c)
        kind = "系数（标准化后）"
    else:
        return None
    if len(v) != len(names):
        return None
    out = pd.DataFrame({"特征": names, kind: v})
    return out.reindex(out[kind].abs().sort_values(ascending=False).index).reset_index(drop=True)


@dataclass
class ShapResult:
    values: np.ndarray  # (n, f) 对正类（或回归输出）的 SHAP 值
    data: pd.DataFrame  # 变换后的特征值
    names: list[str]
    base: float
    explainer: Any
    pipe: Pipeline
    kind: str
    output: str  # "对数几率" / "概率" / "预测值"

    def mean_abs(self) -> pd.DataFrame:
        m = np.abs(self.values).mean(0)
        return pd.DataFrame({"特征": self.names, "mean|SHAP|": m}).sort_values("mean|SHAP|", ascending=False).reset_index(drop=True)

    def explain(self, X_new: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, float]:
        """对新样本给出 SHAP 值（用于单个患者解释）。"""
        Z = self.pipe.named_steps["prep"].transform(X_new)
        sv = _pos_values(self.explainer.shap_values(_kernel_input(self, Z)))
        return sv, Z, self.base


def _kernel_input(res: "ShapResult", Z: pd.DataFrame):
    return Z.to_numpy() if res.kind == "kernel" else Z


def _pos_values(sv) -> np.ndarray:
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return sv


def compute_shap(pipe: Pipeline, spec: ModelSpec, X: pd.DataFrame, task: str, n_classes: int = 2,
                 max_rows: int = 300, seed: int = 0) -> ShapResult:
    """支持二分类与回归。树模型 TreeExplainer，线性模型 LinearExplainer，其它 KernelExplainer。"""
    import shap

    if task == "classification" and n_classes != 2:
        raise ValueError("SHAP 目前仅支持二分类与回归任务")
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xs = X if len(X) <= max_rows else X.sample(max_rows, random_state=seed)
    Z = prep.transform(Xs)
    names = list(Z.columns)
    kind, expl, output = "kernel", None, "概率" if task == "classification" else "预测值"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec.tree_based and spec.key not in ("adaboost",):
            try:
                expl = shap.TreeExplainer(model)
                kind = "tree"
                output = "对数几率" if task == "classification" and spec.key not in ("rf", "dtree") else output
                sv = _pos_values(expl.shap_values(Z))
            except Exception:
                expl = None
        if expl is None and spec.linear:
            try:
                expl = shap.LinearExplainer(model, Z)
                kind = "linear"
                output = "对数几率" if task == "classification" else "预测值"
                sv = _pos_values(expl.shap_values(Z))
            except Exception:
                expl = None
        if expl is None:
            bg = shap.kmeans(Z.to_numpy(), min(15, len(Z)))
            f = ((lambda a: model.predict_proba(pd.DataFrame(a, columns=names))[:, 1]) if task == "classification"
                 else (lambda a: model.predict(pd.DataFrame(a, columns=names))))
            expl = shap.KernelExplainer(f, bg)
            kind = "kernel"
            sv = _pos_values(expl.shap_values(Z.to_numpy(), nsamples=120, silent=True))
    base = np.asarray(expl.expected_value).reshape(-1)[-1]
    return ShapResult(sv, Z, names, float(base), expl, pipe, kind, output)
