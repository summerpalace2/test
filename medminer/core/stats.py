"""统计分析：描述统计、基线特征表（Table 1）、相关性、单/多因素逻辑回归（OR）。"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .schema import BINARY, CATEGORICAL, NUMERIC


def describe(df: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        t = types.get(c, "")
        row = {"变量": c, "类型": t, "非空": int(s.notna().sum()), "缺失%": round(s.isna().mean() * 100, 2),
               "唯一值": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s) and t in (NUMERIC, BINARY, CATEGORICAL):
            x = s.dropna().astype(float)
            if len(x):
                row.update(均值=x.mean(), 标准差=x.std(), 最小=x.min(), Q1=x.quantile(.25), 中位数=x.median(),
                           Q3=x.quantile(.75), 最大=x.max(),
                           偏度=stats.skew(x) if len(x) > 2 else np.nan,
                           峰度=stats.kurtosis(x) if len(x) > 3 else np.nan)
        else:
            vc = s.value_counts(dropna=True)
            if len(vc):
                row.update(众数=vc.index[0], 众数占比=round(vc.iloc[0] / max(1, n - s.isna().sum()) * 100, 1))
        rows.append(row)
    out = pd.DataFrame(rows)
    num_cols = out.select_dtypes("number").columns
    return out.assign(**{c: out[c].round(3) for c in num_cols})


def normality(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        x = df[c].dropna().astype(float)
        if len(x) < 3:
            continue
        if len(x) > 5000:
            x = x.sample(5000, random_state=0)
        w, p = stats.shapiro(x)
        rows.append({"变量": c, "Shapiro W": round(w, 4), "p 值": p, "近似正态(p>0.05)": "是" if p > 0.05 else "否"})
    return pd.DataFrame(rows)


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table_one(df: pd.DataFrame, group: str, types: dict[str, str], cols: list[str] | None = None,
              group_labels: dict | None = None) -> pd.DataFrame:
    """基线特征表。连续变量：近似正态 → 均值±SD + t 检验/ANOVA；否则 中位数[Q1,Q3] + Mann-Whitney/Kruskal。
    分类变量：n(%) + 卡方（期望频数<5 的 2×2 用 Fisher 精确检验）。"""
    d = df[df[group].notna()]
    levels = sorted(d[group].unique(), key=lambda v: (str(type(v)), v))
    cols = [c for c in (cols or list(df.columns)) if c != group and types.get(c) in (NUMERIC, BINARY, CATEGORICAL)]
    names = {lv: (group_labels or {}).get(lv, f"{group}={lv}") for lv in levels}
    header = ["变量", "总体"] + [f"{names[lv]} (n={int((d[group] == lv).sum())})" for lv in levels] + ["统计量", "检验方法", "p 值"]
    rows = [[f"样本量 n", str(len(d))] + [str(int((d[group] == lv).sum())) for lv in levels] + ["", "", ""]]

    for c in cols:
        s = d[c]
        miss = int(s.isna().sum())
        if types.get(c) == NUMERIC:
            groups = [d.loc[d[group] == lv, c].dropna().astype(float).to_numpy() for lv in levels]
            groups_ok = [g for g in groups if len(g) >= 2]
            normal = bool(groups_ok) and all(
                (stats.shapiro(g[:5000])[1] > 0.05) if len(g) >= 3 and np.ptp(g) > 0 else True for g in groups_ok)
            if normal:
                f = lambda x: f"{np.mean(x):.2f} ± {np.std(x, ddof=1):.2f}" if len(x) > 1 else "-"
                label = f"{c}, 均值±SD"
            else:
                f = lambda x: (f"{np.median(x):.2f} [{np.percentile(x, 25):.2f}, {np.percentile(x, 75):.2f}]"
                               if len(x) else "-")
                label = f"{c}, 中位数[Q1,Q3]"
            stat, p, test = np.nan, np.nan, ""
            if len(groups_ok) >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        if len(groups_ok) == 2:
                            if normal:
                                r = stats.ttest_ind(*groups_ok, equal_var=False)
                                test = "Welch t 检验"
                            else:
                                r = stats.mannwhitneyu(*groups_ok, alternative="two-sided")
                                test = "Mann-Whitney U"
                        else:
                            r = stats.f_oneway(*groups_ok) if normal else stats.kruskal(*groups_ok)
                            test = "单因素 ANOVA" if normal else "Kruskal-Wallis"
                        stat, p = float(r.statistic), float(r.pvalue)
                    except ValueError:
                        pass
            tot = s.dropna().astype(float).to_numpy()
            rows.append([label + (f"（缺失{miss}）" if miss else ""), f(tot)] + [f(g) for g in groups]
                        + ["" if np.isnan(stat) else f"{stat:.3f}", test, fmt_p(p)])
        else:
            cats = sorted(s.dropna().unique(), key=lambda v: (str(type(v)), v))
            ct = pd.crosstab(s, d[group]).reindex(index=cats, columns=levels, fill_value=0)
            stat, p, test = np.nan, np.nan, ""
            if ct.shape[0] >= 2 and ct.shape[1] >= 2 and ct.to_numpy().sum() > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    chi2, p_chi, _, exp = stats.chi2_contingency(ct.to_numpy(), correction=False)
                    if ct.shape == (2, 2) and (exp < 5).any():
                        _, p = stats.fisher_exact(ct.to_numpy())
                        test = "Fisher 精确"
                    else:
                        stat, p, test = float(chi2), float(p_chi), "卡方检验"
            rows.append([f"{c}" + (f"（缺失{miss}）" if miss else ""), "", *[""] * len(levels),
                         "" if np.isnan(stat) else f"{stat:.3f}", test, fmt_p(p)])
            for cat in cats:
                tot_n = int((s == cat).sum())
                cells = []
                for lv in levels:
                    n_lv = int(((s == cat) & (d[group] == lv)).sum())
                    denom = int(((d[group] == lv) & s.notna()).sum())
                    cells.append(f"{n_lv} ({n_lv / denom * 100:.1f}%)" if denom else "-")
                rows.append([f"    {cat:g}" if isinstance(cat, (int, float, np.number)) else f"    {cat}",
                             f"{tot_n} ({tot_n / max(1, int(s.notna().sum())) * 100:.1f}%)", *cells, "", "", ""])
    return pd.DataFrame(rows, columns=header)


def correlation(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)


def target_association(df: pd.DataFrame, target: str, types: dict[str, str]) -> pd.DataFrame:
    """各变量与目标的关联强度：数值-二分类用点二列/AUC，其它用 Cramér V / η。"""
    rows = []
    y = df[target]
    for c in df.columns:
        if c == target or types.get(c) not in (NUMERIC, BINARY, CATEGORICAL):
            continue
        d = df[[c, target]].dropna()
        if d.empty:
            continue
        if types.get(c) == NUMERIC and d[target].nunique() == 2:
            yy = (d[target] == sorted(d[target].unique())[-1]).astype(int)
            r, p = stats.pointbiserialr(yy, d[c].astype(float))
            rows.append({"变量": c, "方法": "点二列相关 r", "关联强度": r, "p 值": p})
        else:
            ct = pd.crosstab(d[c], d[target])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, p, _, _ = stats.chi2_contingency(ct.to_numpy(), correction=False)
                v = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))
                rows.append({"变量": c, "方法": "Cramér V", "关联强度": v, "p 值": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.reindex(out["关联强度"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return out


def logit_or(df: pd.DataFrame, target: str, features: list[str], types: dict[str, str],
             multivariable: bool = False, standardize: bool = True) -> pd.DataFrame:
    """逻辑回归 OR（比值比）。target 需为二分类。连续变量默认按 1 个标准差报告 OR；分类变量与第一个水平比较。"""
    import statsmodels.api as sm

    d = df[[target] + features].dropna().copy()
    lv = sorted(d[target].unique())
    if len(lv) != 2:
        raise ValueError("OR 分析要求目标列为二分类（请先在预处理中二值化）")
    y = (d[target] == lv[1]).astype(int)

    def design(cols):
        parts = []
        for c in cols:
            if types.get(c) == NUMERIC:
                x = d[c].astype(float)
                if standardize and x.std() > 0:
                    x = (x - x.mean()) / x.std()
                parts.append(x.rename(c))
            else:
                dm = pd.get_dummies(d[c].astype(str), prefix=c, prefix_sep="=", drop_first=True, dtype=float)
                parts.append(dm)
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=d.index)

    def fit(X):
        X = sm.add_constant(X, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return sm.Logit(y, X).fit(disp=0, maxiter=200)
            except Exception:
                return None

    rows = []
    groups = [features] if multivariable else [[c] for c in features]
    for g in groups:
        X = design(g)
        if X.shape[1] == 0:
            continue
        res = fit(X)
        if res is None:
            continue
        ci = res.conf_int()
        for name in X.columns:
            rows.append({"变量": name, "OR": float(np.exp(res.params[name])),
                         "95%CI下限": float(np.exp(ci.loc[name, 0])), "95%CI上限": float(np.exp(ci.loc[name, 1])),
                         "p 值": float(res.pvalues[name]),
                         "分析": "多因素" if multivariable else "单因素"})
    return pd.DataFrame(rows)
