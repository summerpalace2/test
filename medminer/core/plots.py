"""所有图表的绘制函数。统一约定：fn(fig, ...) 在传入的 matplotlib Figure 上作图（不使用 pyplot，线程安全），
界面和报告导出共用。"""
from __future__ import annotations

import io
import warnings

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.figure import Figure

from .schema import BINARY, CATEGORICAL, NUMERIC

PALETTE = ["#2F6FDE", "#E8603C", "#2BA89B", "#F2A33A", "#8A5CC2", "#D1477A", "#6B7A90", "#4CAF50", "#A0522D", "#17A2B8"]
GOOD, BAD, NEUTRAL = "#2BA89B", "#E8603C", "#8A94A6"

_CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Noto Sans CJK SC", "Source Han Sans SC",
                   "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Arial Unicode MS", "SimSun"]


def setup_fonts() -> str | None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in _CJK_CANDIDATES if n in names), None)
    base = ["DejaVu Sans"]
    matplotlib.rcParams["font.sans-serif"] = ([chosen] if chosen else []) + base
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10, "figure.dpi": 100,
        "savefig.dpi": 160, "legend.frameon": False,
    })
    sns.set_palette(PALETTE)
    return chosen


setup_fonts()


def new_figure(w: float = 6.4, h: float = 4.4) -> Figure:
    return Figure(figsize=(w, h), layout="constrained")


def fig_to_png(fig: Figure, dpi: int = 160) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    return buf.getvalue()


def _clear(fig: Figure):
    fig.clear()
    return fig


def _empty(fig, msg):
    _clear(fig)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=NEUTRAL, fontsize=12, wrap=True)
    return fig


# ===================== 通用数据可视化 =====================
CHART_KINDS = {
    "histogram": ("直方图", dict(x="num")),
    "kde": ("密度曲线", dict(x="num")),
    "box": ("箱线图", dict(y="num", x="cat?")),
    "violin": ("小提琴图", dict(y="num", x="cat?")),
    "strip": ("分簇散点（蜂群）", dict(y="num", x="cat")),
    "scatter": ("散点图", dict(x="num", y="num", hue="any?")),
    "regplot": ("散点+回归线", dict(x="num", y="num", hue="cat?")),
    "count": ("计数柱状图", dict(x="cat", hue="cat?")),
    "stacked": ("构成比（堆叠百分比）", dict(x="cat", hue="cat")),
    "bar_mean": ("分组均值±95%CI", dict(x="cat", y="num")),
    "pie": ("饼图", dict(x="cat")),
    "line": ("折线图（按行序/X）", dict(x="any?", y="num")),
    "corr": ("相关性热力图", dict()),
    "pairs": ("散点矩阵", dict(hue="cat?")),
    "missing": ("缺失值分布图", dict()),
}


def draw_chart(fig: Figure, kind: str, df: pd.DataFrame, types: dict[str, str], x: str | None = None, y: str | None = None,
               hue: str | None = None, cols: list[str] | None = None, bins: int = 20, corr_method: str = "pearson",
               annotate: bool = True) -> Figure:
    _clear(fig)
    hue = hue or None
    d = df
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind == "corr":
                return _corr(fig, d, types, cols, corr_method, annotate)
            if kind == "pairs":
                return _pairs(fig, d, types, cols, hue)
            if kind == "missing":
                return _missing(fig, d)
            ax = fig.add_subplot(111)
            hue_ok = hue if hue and hue in d.columns else None
            data = d.copy()
            if hue_ok and types.get(hue_ok) != NUMERIC:
                # 分组变量转为“有序分类”：图例与分组顺序按取值大小排列，而不是出现顺序
                fmt = lambda v: f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)
                uniq = sorted(d[hue_ok].dropna().unique(), key=lambda v: (str(type(v)), v))
                data[hue_ok] = pd.Categorical(d[hue_ok].map(lambda v: fmt(v) if pd.notna(v) else v),
                                              categories=[fmt(u) for u in uniq], ordered=True)
            if kind == "histogram":
                sns.histplot(data=data, x=x, hue=hue_ok, bins=bins, kde=True, ax=ax, multiple="layer" if hue_ok else "layer",
                             alpha=0.55)
                ax.set_title(f"{x} 的分布")
            elif kind == "kde":
                sns.kdeplot(data=data, x=x, hue=hue_ok, fill=True, common_norm=False, alpha=0.4, ax=ax)
                ax.set_title(f"{x} 的密度曲线")
            elif kind in ("box", "violin", "strip"):
                data[x] = data[x].astype(str) if x else None
                fn = {"box": sns.boxplot, "violin": sns.violinplot, "strip": sns.stripplot}[kind]
                kw = dict(data=data, y=y, ax=ax)
                if x:
                    order = sorted(data[x].dropna().unique(), key=lambda v: (len(v), v))
                    kw.update(x=x, order=order, hue=x, legend=False)
                if kind == "strip":
                    kw.update(alpha=0.65, size=4)
                if kind == "box" and x:
                    kw.update(showfliers=True)
                fn(**kw)
                if kind == "box" and x:
                    sns.stripplot(data=data, x=x, y=y, order=order, color="black", alpha=0.25, size=2.5, ax=ax)
                ax.set_title(f"{y}" + (f" 按 {x} 分组" if x else ""))
            elif kind == "scatter":
                sns.scatterplot(data=data, x=x, y=y, hue=hue_ok, ax=ax, alpha=0.75, s=36)
                ax.set_title(f"{y} vs {x}")
            elif kind == "regplot":
                if hue_ok:
                    for i, (k, g) in enumerate(data.groupby(hue_ok)):
                        sns.regplot(data=g, x=x, y=y, ax=ax, label=str(k), color=PALETTE[i % len(PALETTE)],
                                    scatter_kws=dict(alpha=0.5, s=26), line_kws=dict(lw=2))
                    ax.legend(title=hue_ok)
                else:
                    sns.regplot(data=data, x=x, y=y, ax=ax, scatter_kws=dict(alpha=0.5, s=30), line_kws=dict(color=BAD, lw=2))
                    ok = data[[x, y]].dropna()
                    if len(ok) > 2:
                        r = np.corrcoef(ok[x].astype(float), ok[y].astype(float))[0, 1]
                        ax.text(0.03, 0.95, f"r = {r:.3f}  (n={len(ok)})", transform=ax.transAxes, va="top")
                ax.set_title(f"{y} vs {x}")
            elif kind == "count":
                data[x] = data[x].astype(str)
                order = sorted(data[x].dropna().unique(), key=lambda v: (len(v), v))
                sns.countplot(data=data, x=x, hue=hue_ok, order=order, ax=ax)
                if not hue_ok:
                    for c in ax.containers:
                        ax.bar_label(c, padding=2, fontsize=9)
                ax.set_title(f"{x} 计数")
            elif kind == "stacked":
                ct = pd.crosstab(d[x].astype(str), d[hue_ok].astype(str) if hue_ok else d[y].astype(str), normalize="index") * 100
                ct.plot(kind="bar", stacked=True, ax=ax, color=PALETTE[:ct.shape[1]], width=0.75, rot=0)
                ax.set_ylabel("百分比 (%)")
                ax.legend(title=hue_ok or y, bbox_to_anchor=(1.01, 1), loc="upper left")
                ax.set_title(f"{x} 内的 {hue_ok} 构成")
            elif kind == "bar_mean":
                data[x] = data[x].astype(str)
                order = sorted(data[x].dropna().unique(), key=lambda v: (len(v), v))
                sns.barplot(data=data, x=x, y=y, order=order, errorbar=("ci", 95), hue=x, legend=False, ax=ax, capsize=0.12)
                ax.set_title(f"各 {x} 的 {y} 均值（95% CI）")
            elif kind == "pie":
                vc = d[x].astype(str).value_counts()
                if len(vc) > 8:
                    vc = pd.concat([vc.iloc[:7], pd.Series({"其它": vc.iloc[7:].sum()})])
                ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=PALETTE[:len(vc)], startangle=90,
                       wedgeprops=dict(edgecolor="white"))
                ax.grid(False)
                ax.set_title(f"{x} 构成")
            elif kind == "line":
                xs = data[x] if x else pd.Series(range(len(data)), name="行号")
                tmp = pd.DataFrame({"x": xs.to_numpy(), "y": data[y].to_numpy()}).dropna().sort_values("x")
                ax.plot(tmp["x"], tmp["y"], lw=1.6, color=PALETTE[0])
                ax.set_xlabel(x or "行号")
                ax.set_ylabel(y)
                ax.set_title(f"{y} 趋势")
            else:
                return _empty(fig, f"未知图表类型 {kind}")
            return fig
    except Exception as e:
        return _empty(fig, f"无法绘制：{type(e).__name__}\n{e}")


def _corr(fig, d, types, cols, method, annotate):
    cols = [c for c in (cols or list(d.columns)) if pd.api.types.is_numeric_dtype(d[c]) and types.get(c) != "ignore"]
    if len(cols) < 2:
        return _empty(fig, "至少需要两个数值列")
    c = d[cols].astype(float).corr(method=method)
    ax = fig.add_subplot(111)
    n = len(cols)
    sns.heatmap(c, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=annotate and n <= 16, fmt=".2f",
                annot_kws=dict(size=8 if n > 8 else 9), square=True, linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws=dict(shrink=0.8, label=f"{method} 相关系数"))
    ax.grid(False)
    ax.set_title("相关性热力图")
    ax.tick_params(axis="x", rotation=45)
    return fig


def _pairs(fig, d, types, cols, hue):
    cols = [c for c in (cols or []) if pd.api.types.is_numeric_dtype(d[c])][:6]
    if len(cols) < 2:
        cols = [c for c in d.columns if types.get(c) == NUMERIC][:5]
    if len(cols) < 2:
        return _empty(fig, "至少需要两个数值列")
    n = len(cols)
    axes = fig.subplots(n, n)
    g = d[hue].astype(str) if hue and hue in d.columns else None
    levels = sorted(g.dropna().unique(), key=lambda v: (len(v), v)) if g is not None else [None]
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            ax.grid(False)
            for k, lv in enumerate(levels):
                sub = d if lv is None else d[g == lv]
                col = PALETTE[k % len(PALETTE)]
                if i == j:
                    ax.hist(sub[ci].dropna(), bins=15, alpha=0.55, color=col)
                else:
                    ax.scatter(sub[cj], sub[ci], s=7, alpha=0.55, color=col, label=None if lv is None else str(lv))
            if j == 0:
                ax.set_ylabel(ci, fontsize=8)
            else:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(cj, fontsize=8)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=7)
    if g is not None:
        h, l = axes[0, 1].get_legend_handles_labels()
        fig.legend(h, l, title=hue, loc="outside right upper", markerscale=2)
    return fig


def _missing(fig, d):
    ax = fig.add_subplot(111)
    m = d.isna().to_numpy().astype(int)
    if m.sum() == 0:
        return _empty(fig, "数据中没有缺失值 ✓")
    ax.imshow(m, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xticks(range(d.shape[1]))
    ax.set_xticklabels(d.columns, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("样本（行）")
    ax.grid(False)
    ax.set_title("缺失值分布（黑色=缺失）")
    return fig


# ===================== 分类评估 =====================
def plot_roc(fig, items: list[tuple[str, np.ndarray, np.ndarray]], ci: dict | None = None):
    """items: [(name, y, p)]"""
    from .evaluate import roc_points
    _clear(fig)
    ax = fig.add_subplot(111)
    for i, (name, y, p) in enumerate(items):
        fpr, tpr, auc = roc_points(y, p)
        lab = f"{name} (AUC={auc:.3f})"
        if ci and name in ci and "roc_auc" in ci[name]:
            lo, hi = ci[name]["roc_auc"]
            lab = f"{name} (AUC={auc:.3f}, 95%CI {lo:.3f}–{hi:.3f})"
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i % len(PALETTE)], label=lab)
    ax.plot([0, 1], [0, 1], "--", color=NEUTRAL, lw=1)
    ax.set(xlabel="1 - 特异度（假阳性率）", ylabel="灵敏度（真阳性率）", xlim=(0, 1), ylim=(0, 1.02), title="ROC 曲线")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    return fig


def plot_pr(fig, items):
    from .evaluate import pr_points
    _clear(fig)
    ax = fig.add_subplot(111)
    for i, (name, y, p) in enumerate(items):
        prec, rec, ap = pr_points(y, p)
        ax.plot(rec, prec, lw=2, color=PALETTE[i % len(PALETTE)], label=f"{name} (AP={ap:.3f})")
    base = float(np.mean(items[0][1])) if items else 0
    ax.axhline(base, ls="--", color=NEUTRAL, lw=1, label=f"患病率 {base:.2f}")
    ax.set(xlabel="召回率（灵敏度）", ylabel="精确率（PPV）", xlim=(0, 1), ylim=(0, 1.02), title="PR 曲线")
    ax.legend(loc="lower left", fontsize=8)
    return fig


def plot_confusion(fig, cm: np.ndarray, labels: list[str], title: str = "混淆矩阵"):
    _clear(fig)
    ax = fig.add_subplot(111)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j] * 100:.0f}%)", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else "#1d2a44", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="预测", ylabel="真实", title=title)
    ax.grid(False)
    return fig


def plot_calibration(fig, items):
    from .evaluate import calib_points
    _clear(fig)
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], "--", color=NEUTRAL, label="理想校准")
    for i, (name, y, p) in enumerate(items):
        try:
            frac, mean_p = calib_points(y, p, n_bins=6)
            ax.plot(mean_p, frac, "o-", color=PALETTE[i % len(PALETTE)], label=name)
        except ValueError:
            pass
    ax.set(xlabel="预测概率", ylabel="实际阳性比例", title="校准曲线", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    return fig


def plot_threshold(fig, tab: pd.DataFrame, thr: float):
    _clear(fig)
    ax = fig.add_subplot(111)
    for i, c in enumerate(["灵敏度", "特异度", "PPV", "NPV"]):
        ax.plot(tab["阈值"], tab[c], lw=2, color=PALETTE[i], label=c)
    ax.axvline(thr, color="black", ls="--", lw=1)
    ax.text(thr, 0.02, f" 当前阈值 {thr:.2f}", fontsize=9)
    ax.set(xlabel="决策阈值", ylabel="指标值", ylim=(0, 1.02), title="阈值-指标曲线")
    ax.legend(fontsize=8)
    return fig


def plot_dca(fig, items):
    """决策曲线分析（净获益）。"""
    _clear(fig)
    ax = fig.add_subplot(111)
    ths = np.linspace(0.01, 0.95, 95)
    y0 = items[0][1]
    prev = float(np.mean(y0))
    ax.plot(ths, [prev - (1 - prev) * t / (1 - t) for t in ths], color=NEUTRAL, lw=1.2, label="全部干预")
    ax.axhline(0, color="black", lw=1, label="全不干预")
    for i, (name, y, p) in enumerate(items):
        nb = []
        n = len(y)
        for t in ths:
            pred = p >= t
            tp = np.sum(pred & (y == 1))
            fp = np.sum(pred & (y == 0))
            nb.append(tp / n - fp / n * t / (1 - t))
        ax.plot(ths, nb, lw=2, color=PALETTE[i % len(PALETTE)], label=name)
    ax.set(xlabel="阈值概率", ylabel="净获益", title="决策曲线分析 (DCA)", ylim=(-0.1, max(prev + 0.05, 0.1)))
    ax.legend(fontsize=8)
    return fig


def plot_model_compare(fig, tab: pd.DataFrame, metric: str, ci: dict | None = None, label: str | None = None):
    """tab：table_numeric()；横向柱状图，按指标排序。"""
    _clear(fig)
    ax = fig.add_subplot(111)
    if metric not in tab.columns:
        return _empty(fig, "没有该指标")
    from .evaluate import HIGHER_BETTER
    higher = HIGHER_BETTER.get(metric, True)
    t = tab.sort_values(metric, ascending=higher)  # barh 自下而上绘制：最优的排在最上面
    names = t["模型"].tolist()
    vals = t[metric].to_numpy()
    err = None
    if ci:
        lo = [vals[i] - ci.get(n, {}).get(metric, (np.nan, np.nan))[0] for i, n in enumerate(names)]
        hi = [ci.get(n, {}).get(metric, (np.nan, np.nan))[1] - vals[i] for i, n in enumerate(names)]
        if not np.isnan(lo).all():
            err = np.array([np.clip(lo, 0, None), np.clip(hi, 0, None)])
    bars = ax.barh(names, vals, xerr=err, color=PALETTE[0], alpha=0.85, capsize=3, error_kw=dict(lw=1, ecolor="#555"))
    best = int(np.argmax(vals) if higher else np.argmin(vals))
    bars[best].set_color(GOOD)
    for b, v in zip(bars, vals):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.3f}", va="center", fontsize=9)
    ax.set(xlabel=label or metric, title=f"模型对比：{label or metric}")
    ax.grid(axis="y", alpha=0)
    ax.margins(x=0.15)
    return fig


def plot_importance(fig, imp: pd.DataFrame, value_col: str, name_col: str, title: str, top: int = 15, err_col: str | None = None):
    _clear(fig)
    ax = fig.add_subplot(111)
    d = imp.head(top).iloc[::-1]
    colors = [GOOD if v >= 0 else BAD for v in d[value_col]]
    ax.barh(d[name_col], d[value_col], xerr=d[err_col] if err_col else None, color=colors, alpha=0.9, capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title=title, xlabel=value_col)
    ax.grid(axis="y", alpha=0)
    return fig


def plot_shap_summary(fig, shap_res, top: int = 12):
    """蜂群图：横轴 SHAP 值，颜色为特征值（红高蓝低）。"""
    _clear(fig)
    ax = fig.add_subplot(111)
    sv, data, names = shap_res.values, shap_res.data, shap_res.names
    order = np.argsort(np.abs(sv).mean(0))[::-1][:top][::-1]
    rng = np.random.default_rng(0)
    cmap = matplotlib.colormaps["coolwarm"]
    for pos, j in enumerate(order):
        v = data.iloc[:, j].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        norm = np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.full_like(v, 0.5)
        jitter = rng.normal(0, 0.09, len(v))
        ax.scatter(sv[:, j], pos + jitter, c=norm, cmap=cmap, s=12, alpha=0.8, vmin=0, vmax=1, linewidths=0)
    ax.set_yticks(range(len(order)), [names[j] for j in order])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel=f"SHAP 值（单位：{shap_res.output}；>0 推高预测）", title="SHAP 汇总图")
    ax.grid(axis="y", alpha=0)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["低", "高"])
    cb.set_label("特征值")
    return fig


def plot_shap_single(fig, names, values, data_row, base: float, output: str = "", top: int = 10):
    """单个样本的 SHAP 贡献条形图（红=增加风险，蓝=降低）。"""
    _clear(fig)
    ax = fig.add_subplot(111)
    values = np.asarray(values).ravel()
    order = np.argsort(np.abs(values))[::-1][:top][::-1]
    labels = [f"{names[j]} = {data_row[j]:.2f}" for j in order]
    ax.barh(labels, values[order], color=[BAD if v > 0 else "#2F6FDE" for v in values[order]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel=f"SHAP 贡献（单位：{output}；基线 {base:.3f}；红=推高，蓝=拉低）", title="该样本的主要影响因素")
    ax.grid(axis="y", alpha=0)
    return fig


def plot_forest(fig, or_df: pd.DataFrame, title: str = "比值比 (OR) 森林图"):
    _clear(fig)
    ax = fig.add_subplot(111)
    d = or_df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    sig = d["p 值"] < 0.05
    ax.hlines(y, d["95%CI下限"], d["95%CI上限"], color=[BAD if s else NEUTRAL for s in sig], lw=2)
    ax.scatter(d["OR"], y, color=[BAD if s else NEUTRAL for s in sig], zorder=3, s=36)
    ax.axvline(1, color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y, d["变量"])
    ax.set(xlabel="OR（对数坐标，红色 = p<0.05）", title=title)
    ax.grid(axis="y", alpha=0)
    return fig


# ===================== 回归评估 =====================
def plot_pred_vs_true(fig, y, pred, name: str = ""):
    _clear(fig)
    ax = fig.add_subplot(111)
    ax.scatter(y, pred, s=18, alpha=0.6, color=PALETTE[0])
    lo, hi = min(np.min(y), np.min(pred)), max(np.max(y), np.max(pred))
    ax.plot([lo, hi], [lo, hi], "--", color=BAD, lw=1.5)
    ax.set(xlabel="真实值", ylabel="预测值（折外）", title=f"预测 vs 真实 {name}")
    ax.set_aspect("equal", adjustable="datalim")
    return fig


def plot_residuals(fig, y, pred):
    _clear(fig)
    axs = fig.subplots(1, 2)
    r = np.asarray(y) - np.asarray(pred)
    axs[0].scatter(pred, r, s=16, alpha=0.6, color=PALETTE[0])
    axs[0].axhline(0, color=BAD, ls="--")
    axs[0].set(xlabel="预测值", ylabel="残差", title="残差图")
    axs[1].hist(r, bins=20, color=PALETTE[2], alpha=0.8)
    axs[1].set(xlabel="残差", title="残差分布")
    return fig


# ===================== 聚类 =====================
def plot_cluster_scatter(fig, emb: np.ndarray, labels: np.ndarray, var=None, method: str = "PCA", truth=None):
    _clear(fig)
    if truth is not None:
        axs = fig.subplots(1, 2, sharex=True, sharey=True)
    else:
        axs = [fig.add_subplot(111)]
    for k in sorted(set(labels)):
        m = labels == k
        axs[0].scatter(emb[m, 0], emb[m, 1], s=22, alpha=0.75, color=NEUTRAL if k < 0 else PALETTE[k % len(PALETTE)],
                       label="噪声" if k < 0 else f"簇 {k} (n={m.sum()})")
    axs[0].legend(fontsize=8)
    xl = f"{method}1" + (f" ({var[0] * 100:.1f}%)" if var is not None else "")
    yl = f"{method}2" + (f" ({var[1] * 100:.1f}%)" if var is not None else "")
    axs[0].set(xlabel=xl, ylabel=yl, title="聚类结果")
    if truth is not None:
        t = pd.Series(truth).astype(str).to_numpy()
        for i, k in enumerate(sorted(set(t))):
            m = t == k
            axs[1].scatter(emb[m, 0], emb[m, 1], s=22, alpha=0.75, color=PALETTE[i % len(PALETTE)], label=str(k))
        axs[1].legend(fontsize=8, title="真实标签")
        axs[1].set(xlabel=xl, title="真实标签（对照）")
    return fig


def plot_k_sweep(fig, sweep: pd.DataFrame):
    _clear(fig)
    a1, a2 = fig.subplots(1, 2)
    a1.plot(sweep["K"], sweep["惯性(SSE)"], "o-", color=PALETTE[0])
    a1.set(xlabel="簇数 K", ylabel="SSE", title="肘部法")
    a2.plot(sweep["K"], sweep["轮廓系数"], "o-", color=PALETTE[2])
    best = sweep.loc[sweep["轮廓系数"].idxmax()]
    a2.scatter([best["K"]], [best["轮廓系数"]], s=120, facecolors="none", edgecolors=BAD, lw=2)
    a2.set(xlabel="簇数 K", ylabel="轮廓系数", title=f"轮廓系数（最佳 K={int(best['K'])}）")
    for a in (a1, a2):
        a.set_xticks(sweep["K"])
    return fig


def plot_cluster_heatmap(fig, df: pd.DataFrame, labels: np.ndarray, features: list[str], types: dict[str, str]):
    """各簇数值特征均值的 z 分数热图。"""
    _clear(fig)
    ax = fig.add_subplot(111)
    cols = [c for c in features if types.get(c) == NUMERIC]
    if not cols:
        return _empty(fig, "没有数值特征")
    d = df.reset_index(drop=True)[cols].astype(float)
    z = (d - d.mean()) / d.std().replace(0, 1)
    z["簇"] = labels
    m = z.groupby("簇").mean().T
    m.columns = [f"簇{c}" for c in m.columns]
    sns.heatmap(m, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws=dict(label="相对总体的 z 分数"))
    ax.grid(False)
    ax.set_title("簇画像（数值特征）")
    return fig


def plot_pca_variance(fig, tab: pd.DataFrame):
    _clear(fig)
    ax = fig.add_subplot(111)
    ax.bar(tab["主成分"], tab["方差贡献率"], color=PALETTE[0], alpha=0.85)
    ax.plot(tab["主成分"], tab["累计贡献率"], "o-", color=BAD)
    ax.set(ylabel="方差贡献率", title="PCA 方差解释", ylim=(0, 1.05))
    return fig


def plot_single_risk(fig, prob: float, threshold: float):
    """单患者风险条。"""
    _clear(fig)
    ax = fig.add_subplot(111)
    ax.barh([0], [1], color="#E7EBF3", height=0.4)
    ax.barh([0], [prob], color=BAD if prob >= threshold else GOOD, height=0.4)
    ax.axvline(threshold, color="black", lw=1.5, ls="--")
    ax.text(threshold, 0.32, f"阈值 {threshold:.2f}", ha="center", fontsize=9)
    ax.text(min(prob, 0.93), 0, f"{prob * 100:.1f}%", ha="right" if prob > 0.12 else "left", va="center",
            color="white" if prob > 0.12 else "black", fontsize=14, fontweight="bold")
    ax.set(xlim=(0, 1), ylim=(-0.5, 0.5), yticks=[], title="预测风险概率")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    return fig
