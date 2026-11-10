"""模型评估：重复分层 K 折 / 留出法，医学指标（灵敏度、特异度、AUC…），阈值分析，bootstrap 置信区间，调参。"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, explained_variance_score, f1_score, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error, median_absolute_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, RepeatedKFold, RepeatedStratifiedKFold,
                                     StratifiedKFold, KFold, train_test_split)
from sklearn.pipeline import Pipeline

from .features import build_preprocessor
from .medical import binary_confusion_metrics
from .models import ModelSpec, build_estimator

ProgressFn = Callable[[int, int, str], None]


class Cancelled(Exception):
    pass


@dataclass
class EvalConfig:
    task: str = "classification"
    strategy: str = "cv"  # cv | holdout
    n_splits: int = 5
    n_repeats: int = 3
    test_size: float = 0.25
    seed: int = 42
    class_weight: str | None = None  # None | 'balanced'
    impute_num: str = "median"
    scale: str = "auto"  # auto(按算法需要) | standard | minmax | robust | none
    n_boot: int = 300


@dataclass
class ModelResult:
    key: str
    name: str
    params: dict
    task: str
    metrics: dict[str, float] = field(default_factory=dict)  # 各折平均
    metrics_std: dict[str, float] = field(default_factory=dict)  # 各折标准差
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)  # OOF bootstrap 95% CI
    fold_scores: pd.DataFrame | None = None
    oof: np.ndarray | None = None  # 分类：(n,k) 概率；回归：(n,) 预测
    oof_index: np.ndarray | None = None  # OOF 对应的 X 行号（留出法只有测试行）
    pipeline: Pipeline | None = None  # 在全部数据上拟合的最终模型
    seconds: float = 0.0
    error: str = ""


@dataclass
class EvalResult:
    config: EvalConfig
    task: str
    classes: list[str]
    y: np.ndarray  # 与 X 行对齐的全部标签
    feature_names: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    results: list[ModelResult]
    X: pd.DataFrame | None = None

    def get(self, key: str) -> ModelResult | None:
        return next((r for r in self.results if r.key == key), None)

    def y_oof(self, r: ModelResult) -> np.ndarray:
        return self.y[r.oof_index] if r.oof_index is not None else self.y

    def table(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            if r.error:
                rows.append({"模型": r.name, "状态": r.error})
                continue
            row = {"模型": r.name}
            for k, v in r.metrics.items():
                sd = r.metrics_std.get(k, np.nan)
                row[METRIC_ZH.get(k, k)] = f"{v:.3f} ± {sd:.3f}" if not np.isnan(sd) else f"{v:.3f}"
            row["耗时(s)"] = round(r.seconds, 2)
            rows.append(row)
        return pd.DataFrame(rows)

    def table_numeric(self) -> pd.DataFrame:
        rows = [{"模型": r.name, **r.metrics} for r in self.results if not r.error]
        return pd.DataFrame(rows)


METRIC_ZH = {
    "accuracy": "准确率", "balanced_accuracy": "平衡准确率", "precision": "精确率(PPV)", "recall": "灵敏度(召回)",
    "specificity": "特异度", "f1": "F1", "roc_auc": "AUC", "avg_precision": "AP(PR-AUC)", "brier": "Brier",
    "mcc": "MCC", "r2": "R²", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE", "medae": "中位绝对误差",
    "evs": "解释方差",
}
HIGHER_BETTER = {"brier": False, "mae": False, "rmse": False, "mape": False, "medae": False}


def default_metric(task: str) -> str:
    return "roc_auc" if task == "classification" else "r2"


def _scale_for(spec: ModelSpec, scale: str) -> str:
    if scale == "auto":
        return "standard" if spec.needs_scaling else "none"
    return scale


def make_pipeline(spec: ModelSpec, params: dict, num_cols, cat_cols, cfg: EvalConfig) -> Pipeline:
    pre = build_preprocessor(num_cols, cat_cols, cfg.impute_num, _scale_for(spec, cfg.scale))
    est = build_estimator(spec, params, cfg.class_weight, cfg.seed)
    return Pipeline([("prep", pre), ("model", est)])


# ---------------- 指标 ----------------
def classification_metrics(y, proba, n_classes: int, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y)
    out: dict[str, float] = {}
    if n_classes == 2:
        p = proba[:, 1]
        pred = (p >= threshold).astype(int)
        cm = binary_confusion_metrics(y, pred)
        out.update(accuracy=cm["accuracy"], balanced_accuracy=balanced_accuracy_score(y, pred),
                   precision=precision_score(y, pred, zero_division=0), recall=cm["sensitivity"],
                   specificity=cm["specificity"], f1=f1_score(y, pred, zero_division=0), mcc=matthews_corrcoef(y, pred))
        if len(np.unique(y)) == 2:
            out["roc_auc"] = roc_auc_score(y, p)
            out["avg_precision"] = average_precision_score(y, p)
        else:
            out["roc_auc"] = out["avg_precision"] = np.nan
        out["brier"] = brier_score_loss(y, p)
    else:
        pred = proba.argmax(1)
        out.update(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                   precision=precision_score(y, pred, average="macro", zero_division=0),
                   recall=recall_score(y, pred, average="macro", zero_division=0),
                   f1=f1_score(y, pred, average="macro", zero_division=0), mcc=matthews_corrcoef(y, pred))
        try:
            out["roc_auc"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
        except ValueError:
            out["roc_auc"] = np.nan
    return out


def regression_metrics(y, pred) -> dict[str, float]:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    out = dict(r2=r2_score(y, pred), mae=mean_absolute_error(y, pred),
               rmse=float(np.sqrt(mean_squared_error(y, pred))), medae=median_absolute_error(y, pred),
               evs=explained_variance_score(y, pred))
    if np.all(y != 0):
        out["mape"] = float(np.mean(np.abs((y - pred) / y)))
    return out


def bootstrap_ci(y, oof, task: str, n_classes: int, threshold: float, n_boot: int, seed: int,
                 keys: list[str]) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if task == "classification" and len(np.unique(yb)) < (2 if n_classes == 2 else n_classes):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = (classification_metrics(yb, oof[idx], n_classes, threshold) if task == "classification"
                 else regression_metrics(yb, oof[idx]))
        for k in keys:
            if k in m and not np.isnan(m[k]):
                vals[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items() if len(v) > 20}


# ---------------- 评估主流程 ----------------
def _splits(cfg: EvalConfig, X, y):
    if cfg.strategy == "holdout":
        idx = np.arange(len(y))
        strat = y if cfg.task == "classification" and np.min(np.bincount(y)) >= 2 else None
        tr, te = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.seed, stratify=strat)
        return [(tr, te)]
    if cfg.task == "classification":
        k = int(min(cfg.n_splits, np.min(np.bincount(y)))) if len(y) else cfg.n_splits
        k = max(2, k)
        return list(RepeatedStratifiedKFold(n_splits=k, n_repeats=cfg.n_repeats, random_state=cfg.seed).split(X, y))
    return list(RepeatedKFold(n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.seed).split(X, y))


def evaluate_models(X: pd.DataFrame, y: np.ndarray, classes: list[str], num_cols, cat_cols,
                    specs: list[tuple[ModelSpec, dict]], cfg: EvalConfig,
                    progress: ProgressFn | None = None, cancel: Callable[[], bool] | None = None) -> EvalResult:
    n_classes = len(classes)
    splits = _splits(cfg, X, y)
    total = len(specs) * (len(splits) + 1)
    done = 0
    results: list[ModelResult] = []
    keys_ci = (["roc_auc", "accuracy", "recall", "specificity", "f1"] if cfg.task == "classification" and n_classes == 2
               else ["accuracy", "f1"] if cfg.task == "classification" else ["r2", "rmse", "mae"])

    for spec, params in specs:
        res = ModelResult(spec.key, spec.name, params, cfg.task)
        t0 = time.time()
        try:
            pipe = make_pipeline(spec, params, num_cols, cat_cols, cfg)
            width = n_classes if cfg.task == "classification" else 1
            oof_sum = np.zeros((len(y), width))
            oof_cnt = np.zeros(len(y))
            fold_rows = []
            for fi, (tr, te) in enumerate(splits):
                if cancel and cancel():
                    raise Cancelled()
                p = clone(pipe)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p.fit(X.iloc[tr], y[tr])
                    if cfg.task == "classification":
                        proba = np.zeros((len(te), n_classes))
                        proba[:, p.named_steps["model"].classes_.astype(int)] = p.predict_proba(X.iloc[te])
                        fold_rows.append(classification_metrics(y[te], proba, n_classes))
                        oof_sum[te] += proba
                    else:
                        pred = p.predict(X.iloc[te])
                        fold_rows.append(regression_metrics(y[te], pred))
                        oof_sum[te, 0] += pred
                oof_cnt[te] += 1
                done += 1
                if progress:
                    progress(done, total, f"{spec.name}：第 {fi + 1}/{len(splits)} 折")
            mask = oof_cnt > 0
            oof = oof_sum[mask] / oof_cnt[mask][:, None]
            idx = np.where(mask)[0]
            fs = pd.DataFrame(fold_rows)
            res.fold_scores = fs
            res.metrics = fs.mean(numeric_only=True).to_dict()
            res.metrics_std = (fs.std(numeric_only=True, ddof=1).to_dict() if len(fs) > 1
                               else {k: float("nan") for k in fs.columns})
            res.oof = oof if cfg.task == "classification" else oof[:, 0]
            res.oof_index = idx
            res.ci = bootstrap_ci(y[idx], res.oof, cfg.task, n_classes, 0.5, cfg.n_boot, cfg.seed, keys_ci)
            final = clone(pipe)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final.fit(X, y)
            res.pipeline = final
        except Cancelled:
            raise
        except Exception as e:  # 单个模型失败不影响其它模型
            res.error = f"失败：{type(e).__name__}: {e}"
        res.seconds = time.time() - t0
        done = min(total, done + 1)
        if progress:
            progress(done, total, f"{spec.name} 完成")
        results.append(res)
    return EvalResult(cfg, cfg.task, classes, y, list(X.columns), list(num_cols), list(cat_cols), results, X=X)


# ---------------- 阈值 / 曲线 ----------------
def threshold_table(y, p, thresholds=None) -> pd.DataFrame:
    thresholds = np.linspace(0.05, 0.95, 19) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        m = binary_confusion_metrics(y, (p >= t).astype(int))
        rows.append({"阈值": round(float(t), 3), "灵敏度": m["sensitivity"], "特异度": m["specificity"],
                     "PPV": m["ppv"], "NPV": m["npv"], "准确率": m["accuracy"],
                     "TP": m["TP"], "FP": m["FP"], "TN": m["TN"], "FN": m["FN"]})
    return pd.DataFrame(rows)


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def threshold_for_sensitivity(y, p, target: float = 0.9) -> float:
    """满足灵敏度 ≥ target 的最大阈值（筛查场景：优先不漏诊）。"""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]]) if len(ok) else float(thr[-1])


def roc_points(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr, roc_auc_score(y, p)


def pr_points(y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    return prec, rec, average_precision_score(y, p)


def calib_points(y, p, n_bins: int = 6):
    return calibration_curve(y, p, n_bins=n_bins, strategy="quantile")


def conf_matrix(y, pred, labels=None):
    return confusion_matrix(y, pred, labels=labels)


# ---------------- 调参 ----------------
def tune_model(spec: ModelSpec, X, y, num_cols, cat_cols, cfg: EvalConfig, scoring: str | None = None,
               n_iter: int = 20, method: str = "random", params: dict | None = None):
    """返回 (best_params, best_score, cv_results_df)。
    注意：用同一份数据调参并评估会偏乐观，若需无偏估计应对调参后的模型再做独立验证。"""
    grid = spec.grid()
    if not grid:
        raise ValueError(f"{spec.name} 没有可调的超参数网格")
    pipe = make_pipeline(spec, params or {}, num_cols, cat_cols, cfg)
    pgrid = {f"model__{k}": v for k, v in grid.items()}
    if spec.key in ("mlp", "mlp_r"):
        pass
    if cfg.task == "classification":
        k = max(2, min(cfg.n_splits, int(np.min(np.bincount(y)))))
        cv = StratifiedKFold(k, shuffle=True, random_state=cfg.seed)
        scoring = scoring or ("roc_auc" if len(classes_of(y)) == 2 else "accuracy")
    else:
        cv = KFold(cfg.n_splits, shuffle=True, random_state=cfg.seed)
        scoring = scoring or "r2"
    size = int(np.prod([len(v) for v in pgrid.values()]))
    if method == "grid" or size <= n_iter:
        s = GridSearchCV(pipe, pgrid, cv=cv, scoring=scoring, n_jobs=1, error_score=np.nan)
    else:
        s = RandomizedSearchCV(pipe, pgrid, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=1,
                               random_state=cfg.seed, error_score=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.fit(X, y)
    best = {k.replace("model__", ""): v for k, v in s.best_params_.items()}
    cvr = pd.DataFrame(s.cv_results_)
    cols = [c for c in cvr.columns if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "rank_test_score")]
    return best, float(s.best_score_), cvr[cols].sort_values("rank_test_score").reset_index(drop=True)


def classes_of(y):
    return np.unique(y)
