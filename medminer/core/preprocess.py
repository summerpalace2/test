"""可回放的预处理管道。

每个步骤 = (op, params)。步骤分两类：
- fit-free：清洗类（删列/去重/筛选/异常值/二值化…），不从数据“学习”参数，可在全量数据上做；
- fitted：填充/编码/缩放/特征选择，会从数据学习参数。建模页默认在交叉验证的每一折内部重新拟合，
  以避免数据泄漏；这里全量应用只用于探索和可视化。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .params import ParamSpec, defaults
from .schema import CATEGORICAL, NUMERIC, BINARY


@dataclass
class Step:
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict:
        return {"op": self.op, "params": self.params, "enabled": self.enabled}

    @staticmethod
    def from_dict(d: dict) -> "Step":
        return Step(d["op"], d.get("params", {}), d.get("enabled", True))


@dataclass
class Ctx:
    """步骤执行上下文。"""
    types: dict[str, str]
    target: str | None = None


@dataclass
class OpSpec:
    key: str
    label: str
    fitted: bool
    params: list[ParamSpec]
    func: Callable[[pd.DataFrame, dict, Ctx], pd.DataFrame]
    help: str = ""

    def describe(self, p: dict) -> str:
        parts = []
        for s in self.params:
            v = p.get(s.name, s.default)
            if isinstance(v, (list, tuple)):
                v = ",".join(map(str, v)) if len(v) <= 4 else f"{len(v)}列"
            if v in (None, "", []) and s.kind in ("columns", "text"):
                continue  # 空的可选项不显示，描述更简短
            parts.append(f"{s.label.split('（')[0]}={v}")
        return f"{self.label}（{'; '.join(parts)}）" if parts else self.label


def _cols(df: pd.DataFrame, p: dict, key: str = "columns", kind: str | None = None, ctx: Ctx | None = None) -> list[str]:
    cols = [c for c in (p.get(key) or []) if c in df.columns]
    if not cols and kind and ctx is not None:
        if kind == "numeric":
            cols = [c for c in df.columns if ctx.types.get(c) == NUMERIC and c != ctx.target]
        elif kind == "categorical":
            cols = [c for c in df.columns if ctx.types.get(c) in (CATEGORICAL, BINARY) and c != ctx.target]
    return cols


# ---------------- fit-free ----------------
def _drop_columns(df, p, ctx):
    return df.drop(columns=_cols(df, p))


def _drop_duplicates(df, p, ctx):
    return df.drop_duplicates(subset=_cols(df, p) or None).reset_index(drop=True)


def _drop_missing_rows(df, p, ctx):
    cols = _cols(df, p) or list(df.columns)
    ratio = df[cols].isna().mean(axis=1)
    return df[ratio <= float(p.get("max_missing_ratio", 0.0))].reset_index(drop=True)


def _drop_missing_cols(df, p, ctx):
    thr = float(p.get("max_missing_ratio", 0.5))
    keep = [c for c in df.columns if df[c].isna().mean() <= thr or c == ctx.target]
    return df[keep]


def _filter_rows(df, p, ctx):
    col, op, val = p.get("column"), p.get("operator", ">="), p.get("value")
    if col not in df.columns:
        return df
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        v = float(val)
    else:
        v = str(val)
        s = s.astype(str)
    mask = {">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v, "==": s == v, "!=": s != v}[op]
    return df[mask.fillna(False)].reset_index(drop=True)


def _range_filter(df, p, ctx):
    df = df.copy()
    lo, hi, action = p.get("min"), p.get("max"), p.get("action", "to_nan")
    for c in _cols(df, p, kind="numeric", ctx=ctx):
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= df[c] < lo
        if hi is not None:
            bad |= df[c] > hi
        if action == "to_nan":
            df.loc[bad, c] = np.nan
        elif action == "clip":
            df[c] = df[c].clip(lower=lo, upper=hi)
        else:
            df = df[~bad]
    return df.reset_index(drop=True)


def _outliers(df, p, ctx):
    df = df.copy()
    cols = _cols(df, p, kind="numeric", ctx=ctx)
    method, action, k = p.get("method", "iqr"), p.get("action", "to_nan"), float(p.get("k", 1.5))
    mask = pd.DataFrame(False, index=df.index, columns=cols)
    bounds = {}
    for c in cols:
        s = df[c].astype(float)
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        elif method == "zscore":
            m, sd = s.mean(), s.std()
            lo, hi = m - k * sd, m + k * sd
        elif method == "mad":  # 修正 z 分数
            med = s.median()
            mad = (s - med).abs().median() or 1e-9
            lo, hi = med - k * mad / 0.6745, med + k * mad / 0.6745
        elif method == "isolation_forest":
            continue
        else:
            raise ValueError(method)
        bounds[c] = (lo, hi)
        mask[c] = (s < lo) | (s > hi)
    if method == "isolation_forest" and cols:
        X = df[cols].fillna(df[cols].median())
        pred = IsolationForest(contamination=min(0.5, max(0.001, float(p.get("contamination", 0.05)))),
                               random_state=0).fit_predict(X)
        row_bad = pd.Series(pred == -1, index=df.index)
        if action == "remove":
            return df[~row_bad].reset_index(drop=True)
        for c in cols:  # 其余动作对整行可疑值统一置空
            mask[c] = row_bad
    if action == "remove":
        return df[~mask.any(axis=1)].reset_index(drop=True)
    for c in cols:
        if action == "to_nan":
            df.loc[mask[c], c] = np.nan
        elif action == "clip" and c in bounds:
            df[c] = df[c].clip(*bounds[c])
    return df


def _binarize(df, p, ctx):
    df = df.copy()
    col = p.get("column")
    if col not in df.columns:
        return df
    new = p.get("new_name") or col
    th = float(p.get("threshold", 0))
    out = (df[col] > th).astype(float).where(df[col].notna())
    df[new] = out.astype("Int64") if out.dropna().isin([0, 1]).all() else out
    return df


def _map_values(df, p, ctx):
    df = df.copy()
    col = p.get("column")
    if col not in df.columns:
        return df
    mapping = {}
    for part in str(p.get("mapping", "")).replace("，", ",").split(","):
        if ":" in part:
            a, b = part.split(":", 1)
            a, b = a.strip(), b.strip()
            try:
                a = float(a) if pd.api.types.is_numeric_dtype(df[col]) else a
            except ValueError:
                pass
            mapping[a] = b
    s = df[col]
    df[col] = s.map(lambda v: mapping.get(v, v))
    return df


def _cast_type(df, p, ctx):
    df = df.copy()
    to = p.get("to", "numeric")
    for c in _cols(df, p):
        if to == "numeric":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif to == "string":
            df[c] = df[c].astype(str).where(df[c].notna())
        else:
            df[c] = df[c].astype("category")
    return df


def _transform(df, p, ctx):
    df = df.copy()
    f = {"log1p": lambda s: np.log1p(s.clip(lower=0)), "sqrt": lambda s: np.sqrt(s.clip(lower=0)),
         "square": np.square}[p.get("method", "log1p")]
    for c in _cols(df, p, kind="numeric", ctx=ctx):
        df[c] = f(df[c].astype(float))
    return df


# ---------------- fitted ----------------
def _fill_missing(df, p, ctx):
    df = df.copy()
    cols = [c for c in (_cols(df, p) or list(df.columns)) if df[c].isna().any()]
    st = p.get("strategy", "median")
    if st == "knn":
        num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num:
            allnum = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            filled = KNNImputer(n_neighbors=int(p.get("k", 5))).fit_transform(df[allnum])
            tmp = pd.DataFrame(filled, columns=allnum, index=df.index)
            for c in num:
                df[c] = tmp[c]
        cols = [c for c in cols if c not in num]
        st = "mode"
    for c in cols:
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s)
        if st == "mean" and numeric:
            df[c] = s.fillna(s.mean())
        elif st == "median" and numeric:
            df[c] = s.fillna(s.median())
        elif st == "constant":
            v = p.get("constant", 0)
            df[c] = s.fillna(float(v) if numeric else v)
        elif st == "ffill":
            df[c] = s.ffill()
        elif st == "bfill":
            df[c] = s.bfill()
        elif st == "interpolate" and numeric:
            df[c] = s.interpolate(limit_direction="both")
        else:  # mode（也是非数值列使用 mean/median 时的回退）
            m = s.mode(dropna=True)
            if len(m):
                df[c] = s.fillna(m.iloc[0])
    return df


def _encode(df, p, ctx):
    df = df.copy()
    cols = _cols(df, p, kind="categorical", ctx=ctx)
    method = p.get("method", "onehot")
    if method == "onehot":
        return pd.get_dummies(df, columns=cols, drop_first=bool(p.get("drop_first", False)), dtype=int)
    for c in cols:
        codes, _ = pd.factorize(df[c].astype(object), sort=True)
        df[c] = pd.Series(codes, index=df.index).where(df[c].notna())
    return df


def _scale(df, p, ctx):
    df = df.copy()
    cols = _cols(df, p, kind="numeric", ctx=ctx)
    if not cols:
        return df
    sc = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[p.get("method", "standard")]()
    df[cols] = sc.fit_transform(df[cols].astype(float))
    return df


def _feature_select(df, p, ctx):
    method = p.get("method", "variance")
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != ctx.target]
    drop: list[str] = []
    if method == "variance":
        thr = float(p.get("threshold", 0.01))
        drop = [c for c in num if df[c].var() <= thr]
    elif method == "correlation":
        thr = float(p.get("threshold", 0.9))
        corr = df[num].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if (upper[c] > thr).any()]
    return df.drop(columns=drop)


def _op_list() -> list[OpSpec]:
    cols = lambda label="列（不选=按类型自动）": ParamSpec("columns", label, "columns", [])
    return [
        OpSpec("drop_columns", "删除列", False, [cols("要删除的列")], _drop_columns),
        OpSpec("drop_duplicates", "删除重复行", False, [cols("依据列（不选=整行）")], _drop_duplicates),
        OpSpec("drop_missing_rows", "删除缺失行", False,
               [cols("检查列（不选=全部）"),
                ParamSpec("max_missing_ratio", "允许的最大缺失比例", "float", 0.0, 0, 1, 0.05)], _drop_missing_rows),
        OpSpec("drop_missing_cols", "删除高缺失列", False,
               [ParamSpec("max_missing_ratio", "缺失比例超过则删除", "float", 0.5, 0, 1, 0.05)], _drop_missing_cols),
        OpSpec("filter_rows", "按条件筛选行（纳入/排除标准）", False,
               [ParamSpec("column", "列", "column", None),
                ParamSpec("operator", "条件", "choice", ">=", choices=[">", ">=", "<", "<=", "==", "!="]),
                ParamSpec("value", "值", "text", "0")], _filter_rows),
        OpSpec("range_filter", "临床合理范围处理", False,
               [cols(), ParamSpec("min", "下限", "float", None, -1e9, 1e9, 1, nullable=True),
                ParamSpec("max", "上限", "float", None, -1e9, 1e9, 1, nullable=True),
                ParamSpec("action", "超范围值处理", "choice", "to_nan", choices=["to_nan", "clip", "remove"])],
               _range_filter),
        OpSpec("outliers", "异常值处理", False,
               [cols(),
                ParamSpec("method", "检测方法", "choice", "iqr", choices=["iqr", "zscore", "mad", "isolation_forest"]),
                ParamSpec("k", "阈值系数(IQR倍数/σ倍数)", "float", 1.5, 0.5, 10, 0.5),
                ParamSpec("contamination", "孤立森林异常比例", "float", 0.05, 0.001, 0.5, 0.01),
                ParamSpec("action", "处理方式", "choice", "to_nan", choices=["to_nan", "clip", "remove"])], _outliers),
        OpSpec("binarize", "二值化（如诊断 0 / >0）", False,
               [ParamSpec("column", "列", "column", None),
                ParamSpec("threshold", "大于该值记为 1", "float", 0, -1e9, 1e9, 1),
                ParamSpec("new_name", "新列名（空=覆盖原列）", "text", "")], _binarize),
        OpSpec("map_values", "取值映射/重编码", False,
               [ParamSpec("column", "列", "column", None),
                ParamSpec("mapping", "映射（如 1:男,0:女）", "text", "")], _map_values),
        OpSpec("cast_type", "转换数据类型", False,
               [cols("要转换的列"), ParamSpec("to", "目标类型", "choice", "numeric", choices=["numeric", "string", "category"])],
               _cast_type),
        OpSpec("transform", "数值变换", False,
               [cols(), ParamSpec("method", "方法", "choice", "log1p", choices=["log1p", "sqrt", "square"])], _transform),
        OpSpec("fill_missing", "缺失值填充", True,
               [cols("要填充的列（不选=所有有缺失的列）"),
                ParamSpec("strategy", "策略", "choice", "median",
                          choices=["mean", "median", "mode", "constant", "knn", "ffill", "bfill", "interpolate"]),
                ParamSpec("constant", "常数（constant）", "text", "0"),
                ParamSpec("k", "KNN 邻居数", "int", 5, 1, 50, 1)], _fill_missing),
        OpSpec("encode", "类别编码", True,
               [cols(), ParamSpec("method", "方法", "choice", "onehot", choices=["onehot", "label"]),
                ParamSpec("drop_first", "独热时丢弃首类", "bool", False)], _encode),
        OpSpec("scale", "标准化/归一化", True,
               [cols(), ParamSpec("method", "方法", "choice", "standard", choices=["standard", "minmax", "robust"])],
               _scale),
        OpSpec("feature_select", "特征筛选", True,
               [ParamSpec("method", "方法", "choice", "variance", choices=["variance", "correlation"]),
                ParamSpec("threshold", "阈值（方差下限/相关系数上限）", "float", 0.01, 0, 1, 0.01)], _feature_select),
    ]


OPS: dict[str, OpSpec] = {o.key: o for o in _op_list()}

OP_ZH = {k: o.label for k, o in OPS.items()}


def default_params(op: str) -> dict:
    return defaults(OPS[op].params)


def run_pipeline(df: pd.DataFrame, steps: list[Step], ctx: Ctx, include_fitted: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """依次执行步骤，返回 (结果, 日志)。日志每项含步骤描述、行列数变化、错误。"""
    log = []
    cur = df.copy()
    types = dict(ctx.types)
    for i, st in enumerate(steps):
        spec = OPS.get(st.op)
        entry = {"index": i, "desc": spec.describe(st.params) if spec else st.op, "status": "ok", "msg": ""}
        if spec is None or not st.enabled or (spec.fitted and not include_fitted):
            entry["status"] = "skipped"
            log.append(entry)
            continue
        before = cur.shape
        try:
            from .schema import infer_column_type
            c = Ctx(types=types, target=ctx.target)
            cur = spec.func(cur, {**defaults(spec.params), **st.params}, c)
            for col in cur.columns:  # 新增列补推断类型
                if col not in types:
                    types[col] = infer_column_type(cur[col])
            entry["msg"] = f"{before[0]}×{before[1]} → {cur.shape[0]}×{cur.shape[1]}"
        except Exception as e:  # 单步失败不终止，记录并继续
            entry["status"] = "error"
            entry["msg"] = f"{type(e).__name__}: {e}"
        log.append(entry)
    return cur, log


def missing_overview(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    out = pd.DataFrame({"缺失个数": df.isna().sum(), "缺失比例": (df.isna().mean() * 100).round(2)})
    out.index.name = "列"
    return out.reset_index().assign(总行数=n)


def outlier_overview(df: pd.DataFrame, types: dict[str, str], k: float = 1.5) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        if types.get(c) != NUMERIC:
            continue
        s = df[c].dropna().astype(float)
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        rows.append({"列": c, "IQR下界": round(lo, 3), "IQR上界": round(hi, 3),
                     "异常个数": int(((s < lo) | (s > hi)).sum()), "最小": s.min(), "最大": s.max()})
    return pd.DataFrame(rows)
