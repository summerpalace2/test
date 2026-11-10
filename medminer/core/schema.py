"""列类型推断。"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

NUMERIC, CATEGORICAL, BINARY, TEXT, DATETIME, IGNORE = (
    "numeric", "categorical", "binary", "text", "datetime", "ignore")

TYPE_LABELS = {
    NUMERIC: "连续数值", CATEGORICAL: "分类", BINARY: "二分类",
    TEXT: "文本", DATETIME: "日期时间", IGNORE: "忽略",
}
LABEL_TO_TYPE = {v: k for k, v in TYPE_LABELS.items()}

_DATE_RE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")


def infer_column_type(s: pd.Series, max_cat: int = 10) -> str:
    if pd.api.types.is_bool_dtype(s):
        return BINARY
    if pd.api.types.is_datetime64_any_dtype(s):
        return DATETIME
    nn = s.dropna()
    n = len(nn)
    if n == 0:
        return IGNORE
    nun = nn.nunique()
    if pd.api.types.is_numeric_dtype(s):
        if nun <= 1:
            return IGNORE
        if nun == 2:
            return BINARY
        integer_like = bool(np.all(np.mod(nn.to_numpy(dtype=float), 1) == 0))
        if integer_like and nun <= max_cat and nun / n <= 0.2:
            return CATEGORICAL
        return NUMERIC
    sample = nn.astype(str).head(20)
    if sample.map(lambda v: bool(_DATE_RE.match(v))).mean() > 0.9:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if pd.to_datetime(nn.astype(str), errors="coerce").notna().mean() > 0.9:
                return DATETIME
    if nun == 1:
        return IGNORE
    if nun == 2:
        return BINARY
    if nun <= max_cat * 3 or nun / n < 0.5:
        return CATEGORICAL
    return TEXT


def infer_types(df: pd.DataFrame, max_cat: int = 10) -> dict[str, str]:
    return {c: infer_column_type(df[c], max_cat) for c in df.columns}


def split_columns(df: pd.DataFrame, types: dict[str, str], cols: list[str] | None = None):
    """按类型把列分成 (numeric, categorical) 两组；binary 视为分类。忽略/文本/日期被丢弃。"""
    cols = list(cols) if cols is not None else list(df.columns)
    num = [c for c in cols if types.get(c) == NUMERIC]
    cat = [c for c in cols if types.get(c) in (CATEGORICAL, BINARY)]
    return num, cat
