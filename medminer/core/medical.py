"""医学相关：UCI 心脏病（Cleveland）字段字典、临床合理范围、常用指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd

HEART_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# 列 -> (中文名, 说明, 取值含义)
HEART_INFO: dict[str, tuple[str, str, dict | None]] = {
    "age": ("年龄", "单位：岁", None),
    "sex": ("性别", "1=男，0=女", {0: "女", 1: "男"}),
    "cp": ("胸痛类型", "1=典型心绞痛 2=非典型心绞痛 3=非心绞痛性疼痛 4=无症状",
           {1: "典型心绞痛", 2: "非典型心绞痛", 3: "非心绞痛痛", 4: "无症状"}),
    "trestbps": ("静息血压", "入院时，单位 mm Hg", None),
    "chol": ("血清胆固醇", "单位 mg/dl", None),
    "fbs": ("空腹血糖>120mg/dl", "1=是，0=否", {0: "否", 1: "是"}),
    "restecg": ("静息心电图", "0=正常 1=ST-T 异常 2=左心室肥厚",
                {0: "正常", 1: "ST-T异常", 2: "左室肥厚"}),
    "thalach": ("最大心率", "运动试验达到的最大心率", None),
    "exang": ("运动诱发心绞痛", "1=是，0=否", {0: "否", 1: "是"}),
    "oldpeak": ("运动ST段压低", "运动相对静息引起的 ST 段压低", None),
    "slope": ("ST段斜率", "1=上升 2=平坦 3=下降", {1: "上升", 2: "平坦", 3: "下降"}),
    "ca": ("荧光造影主血管数", "0–3 条被染色的主要血管", None),
    "thal": ("地中海贫血/灌注", "3=正常 6=固定缺损 7=可逆缺损", {3: "正常", 6: "固定缺损", 7: "可逆缺损"}),
    "num": ("心脏病诊断", "0=无；1–4=有（血管狭窄>50%）", {0: "无", 1: "有(1)", 2: "有(2)", 3: "有(3)", 4: "有(4)"}),
}

# 临床合理范围（超出的值提示为可疑，而不是直接删除）
CLINICAL_RANGES = {
    "age": (18, 100), "trestbps": (70, 220), "chol": (100, 600),
    "thalach": (60, 220), "oldpeak": (0, 7),
}


def is_heart_dataset(df: pd.DataFrame) -> bool:
    return len(set(HEART_COLUMNS) & set(df.columns)) >= 10


def default_descriptions(df: pd.DataFrame) -> dict[str, str]:
    """列名 -> “中文名：说明”。仅对已知字段给出。"""
    out = {}
    for c in df.columns:
        if c in HEART_INFO and is_heart_dataset(df):
            zh, desc, _ = HEART_INFO[c]
            out[c] = f"{zh}：{desc}"
    return out


def value_labels(df: pd.DataFrame, col: str) -> dict:
    """已知字段的取值含义（如 sex: 0→女, 1→男），未知返回空字典。"""
    if is_heart_dataset(df) and col in HEART_INFO and HEART_INFO[col][2]:
        return dict(HEART_INFO[col][2])
    return {}


def check_clinical_ranges(df: pd.DataFrame, ranges: dict | None = None) -> pd.DataFrame:
    ranges = ranges or CLINICAL_RANGES
    rows = []
    for c, (lo, hi) in ranges.items():
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            bad = int(((s < lo) | (s > hi)).sum())
            rows.append({"变量": c, "合理范围": f"{lo} ~ {hi}", "实际范围": f"{s.min():g} ~ {s.max():g}",
                         "超范围个数": bad})
    return pd.DataFrame(rows)


def binary_confusion_metrics(y_true, y_pred) -> dict[str, float]:
    """灵敏度/特异度/PPV/NPV。y 为 0/1。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def div(a, b):
        return a / b if b else float("nan")

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "accuracy": div(tp + tn, len(y_true))}
