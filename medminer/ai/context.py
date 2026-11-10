"""把分析结果整理成发给大模型的“统计摘要”。只发汇总统计，不发逐行数据；可选隐藏列名。"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..core.schema import BINARY, CATEGORICAL, NUMERIC, TEXT, TYPE_LABELS


class Anonymizer:
    """列名 <-> V1、V2… 的双向映射。关闭时原样返回。"""

    def __init__(self, columns: list[str] | None = None, enabled: bool = False):
        self.enabled = enabled
        self.map = {c: f"V{i + 1}" for i, c in enumerate(columns or [])} if enabled else {}
        self.back = {v: k for k, v in self.map.items()}

    def n(self, col: str) -> str:
        return self.map.get(col, col) if self.enabled else col

    def mask(self, text: str) -> str:
        """把文本中出现的真实列名替换为代号（用于步骤描述等自由文本）。"""
        if not self.enabled or not self.map:
            return text
        for col in sorted(self.map, key=len, reverse=True):
            text = re.sub(rf"(?<![\w]){re.escape(col)}(?![\w])", self.map[col], text)
        return text

    def restore(self, text: str) -> str:
        if not self.enabled or not self.back:
            return text
        pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, self.back), key=len, reverse=True)) + r")\b")
        return pat.sub(lambda m: self.back[m.group(1)], text)

    def legend(self) -> str:
        return "\n".join(f"{v} = {k}" for k, v in self.back.items())


def md_table(df: pd.DataFrame, max_rows: int = 40, index: bool = False) -> str:
    d = df.head(max_rows)
    if index:
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.3f}".rstrip("0").rstrip(".")
        return str(v).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    if len(df) > max_rows:
        lines.append(f"（仅显示前 {max_rows} 行，共 {len(df)} 行）")
    return "\n".join(lines)


def dataset_profile(df: pd.DataFrame, types: dict[str, str], descriptions: dict[str, str], target: str | None,
                    anon: Anonymizer, send_desc: bool = True, max_corr: int = 8) -> str:
    n = len(df)
    out = [f"## 数据概况\n样本数 {n}，变量数 {df.shape[1]}，整体缺失率 {df.isna().mean().mean() * 100:.2f}%。"
           f"重复行 {int(df.duplicated().sum())} 行。"]
    rows = []
    for c in df.columns:
        s, t = df[c], types.get(c, "")
        row = {"变量": anon.n(c), "类型": TYPE_LABELS.get(t, t), "缺失%": round(s.isna().mean() * 100, 1),
               "唯一值": int(s.nunique())}
        if send_desc and descriptions.get(c) and not anon.enabled:
            row["说明"] = descriptions[c]
        if t == NUMERIC:
            x = s.dropna().astype(float)
            if len(x):
                row["取值摘要"] = (f"均值{x.mean():.2f}, SD{x.std():.2f}, 中位{x.median():.2f}, "
                                  f"范围[{x.min():.2f}, {x.max():.2f}], 偏度{x.skew():.2f}")
        elif t in (CATEGORICAL, BINARY):
            vc = s.value_counts(dropna=True).head(8)
            row["取值摘要"] = "; ".join(f"{k}:{v}" for k, v in vc.items())
        elif t == TEXT:
            row["取值摘要"] = "（文本列，内容未发送）"
        rows.append(row)
    out.append(md_table(pd.DataFrame(rows), 80))
    if target and target in df.columns:
        vc = df[target].value_counts(dropna=False)
        out.append(f"\n目标变量：{anon.n(target)}；分布：" + "; ".join(f"{k}:{v}({v / n * 100:.1f}%)" for k, v in vc.items()))
    num = [c for c in df.columns if types.get(c) == NUMERIC]
    if len(num) >= 2:
        c = df[num].corr().abs()
        pairs = [(a, b, df[num].corr().loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:]]
        pairs.sort(key=lambda p: -abs(p[2]) if not np.isnan(p[2]) else 0)
        out.append("\n相关性最强的数值变量对（Pearson）：" + "; ".join(
            f"{anon.n(a)}-{anon.n(b)}: {r:.2f}" for a, b, r in pairs[:max_corr]))
    return "\n".join(out)


def preprocess_summary(log: list[dict], anon: Anonymizer) -> str:
    if not log:
        return "## 预处理步骤\n（尚未进行任何预处理）"
    lines = ["## 已执行的预处理步骤"]
    for e in log:
        lines.append(f"{e['index'] + 1}. {anon.mask(e['desc'])} —— {e['status']} {anon.mask(e['msg'])}")
    return "\n".join(lines)


def table_one_summary(tab: pd.DataFrame) -> str:
    return "## 基线特征表（分组比较）\n" + md_table(tab, 60)


def or_summary(tab: pd.DataFrame, anon: Anonymizer) -> str:
    t = tab.copy()
    t["变量"] = t["变量"].map(lambda v: _anon_level(v, anon))
    return "## 逻辑回归比值比（OR）\n连续变量的 OR 为每增加 1 个标准差。\n" + md_table(t, 60)


def _anon_level(name: str, anon: Anonymizer) -> str:
    if "=" in name:
        c, lv = name.split("=", 1)
        return f"{anon.n(c)}={lv}"
    return anon.n(name)


def model_summary(evres, anon: Anonymizer, importance: pd.DataFrame | None = None, imp_name_col: str = "变量",
                  best_key: str | None = None, threshold_info: str = "") -> str:
    cfg = evres.config
    n = len(evres.y)
    strat = (f"{cfg.n_splits} 折 × {cfg.n_repeats} 次重复分层交叉验证" if cfg.strategy == "cv"
             else f"留出法（测试集 {cfg.test_size:.0%}）")
    out = [f"## 模型评估结果\n任务：{'分类' if evres.task == 'classification' else '回归'}；样本数 {n}；验证方式：{strat}；"
           f"类别权重：{cfg.class_weight or '无'}；缺失填充：{cfg.impute_num}（均在折内拟合，无数据泄漏）。"]
    if evres.task == "classification":
        cnt = pd.Series(evres.y).value_counts().sort_index()
        out.append("类别分布：" + "; ".join(f"{evres.classes[int(k)]}:{v}" for k, v in cnt.items()))
    out.append("指标为各折均值 ± 标准差；CI 为折外预测的 bootstrap 95% 置信区间。\n")
    out.append(md_table(evres.table(), 30))
    cis = []
    for r in evres.results:
        if r.ci:
            cis.append(f"{r.name}: " + ", ".join(f"{k} [{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in r.ci.items()))
    if cis:
        out.append("\n置信区间：\n" + "\n".join(cis))
    if threshold_info:
        out.append("\n" + threshold_info)
    if importance is not None and len(importance):
        d = importance.head(10).copy()
        d[imp_name_col] = d[imp_name_col].map(lambda v: _anon_feature(v, anon))
        out.append("\n特征重要性（前 10）：\n" + md_table(d, 10))
    return "\n".join(out)


def _anon_feature(name, anon: Anonymizer) -> str:
    """独热后的特征名形如 cp_4，按最长前缀匹配原列名。"""
    name = str(name)
    if not anon.enabled:
        return name
    for col in sorted(anon.map, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return anon.map[col] + name[len(col):]
    return name


def cluster_summary(res, profile: pd.DataFrame, anon: Anonymizer, crosstab: pd.DataFrame | None = None,
                    truth_metrics: dict | None = None) -> str:
    p = profile.copy()
    p["变量"] = p["变量"].map(anon.n)
    out = [f"## 聚类结果\n算法：{res.method}；参数：{res.params}；簇数 {res.n_clusters}；噪声点 {res.n_noise}；"
           f"各簇样本数：{res.sizes}。",
           "内部指标：" + ", ".join(f"{k}={v:.3f}" for k, v in res.metrics.items()),
           "\n各簇特征画像（数值=均值及相对总体的标准差倍数；分类=众数及占比）：\n" + md_table(p, 40)]
    if crosstab is not None:
        out.append("\n与真实标签的交叉表：\n" + md_table(crosstab, 20, index=True))
    if truth_metrics:
        out.append("一致性指标：" + ", ".join(f"{k}={v:.3f}" for k, v in truth_metrics.items()))
    return "\n".join(out)


def patient_summary(values: dict, prob: float, threshold: float, contrib: pd.DataFrame, anon: Anonymizer,
                    model_name: str) -> str:
    c = contrib.copy()
    c["特征"] = c["特征"].map(lambda v: _anon_feature(v, anon))
    vals = "; ".join(f"{anon.n(k)}={v}" for k, v in values.items())
    return (f"## 单个样本预测\n模型：{model_name}；输入：{vals}\n预测阳性概率 {prob:.3f}；决策阈值 {threshold:.2f}；"
            f"判定：{'高风险（≥阈值）' if prob >= threshold else '低风险（<阈值）'}。\n"
            "主要影响因素（SHAP，正值=推高风险）：\n" + md_table(c, 10))


def estimate_tokens(text: str) -> int:
    """粗略估计：中文约 1 字≈1 token，英文约 4 字符≈1 token。"""
    cjk = len(re.findall(r"[一-鿿]", text))
    return cjk + (len(text) - cjk) // 4
