"""聚类分析：运行、选 K（肘部/轮廓系数）、降维坐标、簇画像、与真实标签对照。"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)

from .features import build_preprocessor
from .models import ModelSpec, build_estimator
from .schema import BINARY, CATEGORICAL, NUMERIC


@dataclass
class ClusterResult:
    method: str
    params: dict
    labels: np.ndarray
    features: list[str]
    Xt: pd.DataFrame  # 预处理后的特征矩阵
    metrics: dict[str, float] = field(default_factory=dict)
    n_clusters: int = 0
    n_noise: int = 0
    sizes: dict[int, int] = field(default_factory=dict)


def prepare_matrix(df: pd.DataFrame, features: list[str], types: dict[str, str], scale: str = "standard") -> pd.DataFrame:
    num = [c for c in features if types.get(c) == NUMERIC]
    cat = [c for c in features if types.get(c) in (CATEGORICAL, BINARY)]
    if not num and not cat:
        raise ValueError("没有可用于聚类的数值/分类特征")
    pre = build_preprocessor(num, cat, "median", scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pre.fit_transform(df[num + cat])


def _fit_labels(spec: ModelSpec, params: dict, X: np.ndarray, seed: int) -> np.ndarray:
    est = build_estimator(spec, params, None, seed)
    if hasattr(est, "predict") and spec.key == "gmm":
        est.fit(X)
        return est.predict(X)
    return est.fit_predict(X)


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    mask = labels >= 0
    k = len(set(labels[mask]))
    if k < 2 or mask.sum() <= k:
        return {}
    Xm, lm = X[mask], labels[mask]
    return {"silhouette": float(silhouette_score(Xm, lm)),
            "calinski_harabasz": float(calinski_harabasz_score(Xm, lm)),
            "davies_bouldin": float(davies_bouldin_score(Xm, lm))}


def run_clustering(df: pd.DataFrame, features: list[str], types: dict[str, str], spec: ModelSpec, params: dict,
                   scale: str = "standard", seed: int = 42) -> ClusterResult:
    Xt = prepare_matrix(df, features, types, scale)
    labels = _fit_labels(spec, params, Xt.to_numpy(), seed)
    met = internal_metrics(Xt.to_numpy(), labels)
    uniq, cnt = np.unique(labels, return_counts=True)
    return ClusterResult(spec.key, params, labels, features, Xt, met,
                         n_clusters=int((uniq >= 0).sum()), n_noise=int((labels < 0).sum()),
                         sizes={int(u): int(c) for u, c in zip(uniq, cnt)})


def k_sweep(Xt: pd.DataFrame, k_min: int = 2, k_max: int = 10, seed: int = 42) -> pd.DataFrame:
    X = Xt.to_numpy()
    rows = []
    for k in range(k_min, min(k_max, len(X) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"K": k, "惯性(SSE)": km.inertia_, "轮廓系数": silhouette_score(X, km.labels_),
                     "CH指数": calinski_harabasz_score(X, km.labels_), "DB指数": davies_bouldin_score(X, km.labels_)})
    return pd.DataFrame(rows)


def embed_2d(Xt: pd.DataFrame, method: str = "pca", seed: int = 42):
    X = Xt.to_numpy()
    if method == "tsne" and len(X) > 5:
        emb = TSNE(2, perplexity=min(30, max(5, len(X) // 4)), random_state=seed, init="pca").fit_transform(X)
        return emb, None
    p = PCA(2, random_state=seed).fit(X)
    return p.transform(X), p.explained_variance_ratio_


def pca_summary(Xt: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    p = PCA(min(n, Xt.shape[1], len(Xt)), random_state=0).fit(Xt.to_numpy())
    return pd.DataFrame({"主成分": [f"PC{i + 1}" for i in range(len(p.explained_variance_ratio_))],
                         "方差贡献率": p.explained_variance_ratio_, "累计贡献率": np.cumsum(p.explained_variance_ratio_)})


def cluster_profile(df: pd.DataFrame, labels: np.ndarray, features: list[str], types: dict[str, str]) -> pd.DataFrame:
    """每个簇的特征画像：数值取均值，分类取众数及占比；并给出与总体的差异（z 值）。"""
    d = df.reset_index(drop=True).copy()
    d["_簇"] = labels
    rows = []
    for c in features:
        row = {"变量": c}
        if types.get(c) == NUMERIC:
            tot_mean, tot_sd = d[c].mean(), d[c].std() or 1.0
            row["总体"] = round(tot_mean, 2)
            for k in sorted(d["_簇"].unique()):
                m = d.loc[d["_簇"] == k, c].mean()
                row[f"簇{k}"] = f"{m:.2f} ({(m - tot_mean) / tot_sd:+.2f}σ)"
        else:
            vc = d[c].mode(dropna=True)
            row["总体"] = f"{vc.iloc[0]}" if len(vc) else ""
            for k in sorted(d["_簇"].unique()):
                sub = d.loc[d["_簇"] == k, c].dropna()
                if len(sub):
                    top = sub.mode().iloc[0]
                    row[f"簇{k}"] = f"{top:g} ({(sub == top).mean() * 100:.0f}%)" if isinstance(top, (int, float, np.number)) else f"{top} ({(sub == top).mean() * 100:.0f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def compare_with_truth(labels: np.ndarray, truth: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    t = truth.reset_index(drop=True)
    ok = t.notna().to_numpy()
    ct = pd.crosstab(pd.Series(labels[ok], name="簇"), t[ok].rename("真实标签"))
    met = {"ARI（调整兰德指数）": float(adjusted_rand_score(t[ok].astype(str), labels[ok])),
           "NMI（归一化互信息）": float(normalized_mutual_info_score(t[ok].astype(str), labels[ok]))}
    return ct, met
