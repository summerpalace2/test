import numpy as np
import pandas as pd
import pytest

from medminer.core import cluster as cl
from medminer.core import evaluate as ev
from medminer.core import explain as ex
from medminer.core import plots as pl
from medminer.core import stats
from medminer.core.dataset import Dataset
from medminer.core.features import prepare_xy
from medminer.core.models import REGISTRY
from medminer.core.preprocess import Step


def png_ok(fig, min_bytes=8000):
    data = pl.fig_to_png(fig, dpi=80)
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > min_bytes
    return data


def has_error_text(fig) -> bool:
    return any("无法绘制" in t.get_text() for ax in fig.axes for t in ax.texts)


@pytest.fixture(scope="module")
def ds(heart_df):
    d = Dataset.from_frame("h", heart_df.copy())
    d.add_step(Step("binarize", {"column": "num", "threshold": 0, "new_name": "dx"}))
    return d


KIND_ARGS = {
    "histogram": dict(x="age"), "kde": dict(x="chol", hue="sex"),
    "box": dict(y="chol", x="dx"), "violin": dict(y="thalach", x="cp"), "strip": dict(y="age", x="dx"),
    "scatter": dict(x="age", y="thalach", hue="dx"), "regplot": dict(x="age", y="thalach"),
    "count": dict(x="cp", hue="dx"), "stacked": dict(x="cp", hue="dx"), "bar_mean": dict(x="dx", y="thalach"),
    "pie": dict(x="cp"), "line": dict(x="age", y="chol"), "corr": dict(cols=["age", "chol", "thalach", "oldpeak"]),
    "pairs": dict(cols=["age", "chol", "thalach"], hue="dx"), "missing": {},
}


@pytest.mark.parametrize("kind", list(pl.CHART_KINDS))
def test_every_chart_kind_renders(kind, ds):
    fig = pl.new_figure()
    pl.draw_chart(fig, kind, ds.current(), ds.current_types(), **KIND_ARGS[kind])
    assert not has_error_text(fig), kind
    png_ok(fig)


def test_chart_error_is_shown_not_raised(ds):
    fig = pl.new_figure()
    pl.draw_chart(fig, "histogram", ds.current(), ds.current_types(), x="不存在的列")
    assert has_error_text(fig)


@pytest.fixture(scope="module")
def evres(ds):
    df, t = ds.base(), ds.base_types()
    X, y, classes, num, cat = prepare_xy(df, "dx", [c for c in df.columns if c not in ("num", "dx")], t, "classification")
    res = ev.evaluate_models(X, y, classes, num, cat, [(REGISTRY["logreg"], {}), (REGISTRY["rf"], {"n_estimators": 60})],
                             ev.EvalConfig(n_splits=3, n_repeats=1, n_boot=60))
    return X, y, res


def test_classification_plots(evres):
    X, y, res = evres
    items = [(r.name, res.y_oof(r), r.oof[:, 1]) for r in res.results]
    ci = {r.name: r.ci for r in res.results}
    for fn in (pl.plot_roc, pl.plot_pr, pl.plot_calibration, pl.plot_dca):
        fig = pl.new_figure()
        fn(fig, items, ci) if fn is pl.plot_roc else fn(fig, items)
        png_ok(fig)
    fig = pl.new_figure()
    pl.plot_confusion(fig, ev.conf_matrix(items[0][1], (items[0][2] >= 0.5).astype(int)), ["无病", "有病"])
    png_ok(fig)
    fig = pl.new_figure()
    pl.plot_threshold(fig, ev.threshold_table(items[0][1], items[0][2]), 0.5)
    png_ok(fig)
    fig = pl.new_figure()
    pl.plot_model_compare(fig, res.table_numeric(), "roc_auc", ci, "AUC")
    png_ok(fig)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()][-1] == max(
        res.table_numeric().itertuples(), key=lambda r: r.roc_auc).模型  # 最优模型在最上面


def test_explain_and_stats_plots(evres, ds):
    X, y, res = evres
    r = res.results[1]
    sh = ex.compute_shap(r.pipeline, REGISTRY["rf"], X.head(100), "classification")
    fig = pl.new_figure()
    pl.plot_shap_summary(fig, sh)
    png_ok(fig)
    sv, Z, base = sh.explain(X.iloc[[3]])
    fig = pl.new_figure()
    pl.plot_shap_single(fig, sh.names, sv[0], Z.iloc[0].to_numpy(), base, sh.output)
    png_ok(fig)
    imp = ex.heldout_permutation_importance(X, y, REGISTRY["rf"], {"n_estimators": 40}, res.num_cols, res.cat_cols,
                                            ev.EvalConfig(n_splits=3), n_repeats=3, n_splits=3)
    fig = pl.new_figure()
    pl.plot_importance(fig, imp, "重要性(得分下降)", "变量", "置换重要性", err_col="标准差(折间)")
    png_ok(fig)
    orr = stats.logit_or(ds.current(), "dx", ["age", "thalach", "sex", "oldpeak"], ds.current_types())
    fig = pl.new_figure()
    pl.plot_forest(fig, orr)
    png_ok(fig)
    fig = pl.new_figure()
    pl.plot_single_risk(fig, 0.73, 0.5)
    png_ok(fig, 5000)


def test_regression_and_cluster_plots(ds):
    rng = np.random.default_rng(0)
    y = rng.normal(size=80)
    p = y + rng.normal(0, 0.3, 80)
    for fn in (pl.plot_pred_vs_true, pl.plot_residuals):
        fig = pl.new_figure()
        fn(fig, y, p)
        png_ok(fig)
    df, t = ds.base(), ds.base_types()
    feats = ["age", "chol", "thalach", "oldpeak", "sex"]
    r = cl.run_clustering(df, feats, t, REGISTRY["kmeans"], {"n_clusters": 3})
    emb, var = cl.embed_2d(r.Xt)
    fig = pl.new_figure()
    pl.plot_cluster_scatter(fig, emb, r.labels, var, "PC", truth=df["num"])
    png_ok(fig)
    fig = pl.new_figure()
    pl.plot_k_sweep(fig, cl.k_sweep(r.Xt, 2, 6))
    png_ok(fig)
    fig = pl.new_figure()
    pl.plot_cluster_heatmap(fig, df, r.labels, feats, t)
    png_ok(fig)
    fig = pl.new_figure()
    pl.plot_pca_variance(fig, cl.pca_summary(r.Xt))
    png_ok(fig, 5000)


def test_cjk_font_available_for_chinese_titles():
    # 没有中文字体时标题会变成方块；至少应有一个候选字体被选中（开发机/CI 无中文字体时跳过）
    chosen = pl.setup_fonts()
    if chosen is None:
        pytest.skip("本机没有可用的中文字体")
    assert chosen in pl._CJK_CANDIDATES
