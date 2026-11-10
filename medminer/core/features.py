"""建模用的折内预处理器（sklearn ColumnTransformer）及 X/y 准备。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   RobustScaler, StandardScaler)

from .schema import BINARY, CATEGORICAL, NUMERIC


def _to_object(X):
    df = pd.DataFrame(X).astype(object)
    return df.where(pd.notna(df), np.nan)


def _names_one_to_one(transformer, input_features):
    return np.asarray(input_features, dtype=object)


def build_preprocessor(num_cols: list[str], cat_cols: list[str], impute_num: str = "median",
                       scale: str = "standard", onehot: bool = True) -> ColumnTransformer:
    """数值：缺失填充(+缩放)；分类：众数填充 + 独热。所有统计量只在训练折内学习。"""
    num_steps = [("imp", SimpleImputer(strategy=impute_num))]
    scaler = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}.get(scale)
    if scaler:
        num_steps.append(("scale", scaler()))
    cat_steps = [
        ("obj", FunctionTransformer(_to_object, feature_names_out=_names_one_to_one)),
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), list(num_cols)))
    if cat_cols:
        transformers.append(("cat", Pipeline(cat_steps), list(cat_cols)))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def prepare_xy(df: pd.DataFrame, target: str, features: list[str], types: dict[str, str], task: str):
    """返回 (X, y, classes, num_cols, cat_cols)。分类任务 y 编码为 0..k-1；回归任务 y 为 float。
    目标缺失的行被丢弃。"""
    d = df[df[target].notna()].reset_index(drop=True)
    feats = [f for f in features if f != target and f in d.columns]
    X = d[feats].copy()
    num_cols = [c for c in feats if types.get(c) == NUMERIC]
    cat_cols = [c for c in feats if types.get(c) in (CATEGORICAL, BINARY)]
    # 其它类型（文本/日期/忽略）不参与建模
    keep = num_cols + cat_cols
    X = X[keep]
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(d[target].astype(str) if not pd.api.types.is_numeric_dtype(d[target]) else d[target])
        classes = [str(c) if not isinstance(c, (np.integer, np.floating)) else f"{c:g}" for c in le.classes_]
    else:
        y = d[target].astype(float).to_numpy()
        classes = []
    return X, y, classes, num_cols, cat_cols
