import numpy as np
import pandas as pd
import pytest

from medminer.core import cluster as cl
from medminer.core import evaluate as ev
from medminer.core import explain as ex
from medminer.core.features import prepare_xy
from medminer.core.models import CLASSIFIERS, CLUSTERERS, REGISTRY, REGRESSORS
from medminer.core.preprocess import Step
from medminer.core.dataset import Dataset


@pytest.fixture(scope="module")
def clf_data(heart_df):
    ds = Dataset.from_frame("h", heart_df.copy())
    ds.add_step(Step("binarize", {"column": "num", "threshold": 0, "new_name": "dx"}))
    df, types = ds.base(), ds.base_types()
    feats = [c for c in df.columns if c not in ("num", "dx")]
    return prepare_xy(df, "dx", feats, types, "classification")


def test_prepare_xy_keeps_nan_for_infold_imputation(clf_data):
    X, y, classes, num, cat = clf_data
    assert classes == ["0", "1"] and set(y) == {0, 1}
    assert X["ca"].isna().sum() == 4  # 缺失保留，交给折内填充
    assert "num" not in X.columns and "dx" not in X.columns


@pytest.mark.parametrize("spec", CLASSIFIERS, ids=lambda s: s.key)
def test_every_classifier_trains_and_beats_chance(spec, clf_data):
    X, y, classes, num, cat = clf_data
    cfg = ev.EvalConfig(n_splits=3, n_repeats=1, n_boot=30)
    res = ev.evaluate_models(X, y, classes, num, cat, [(spec, {})], cfg)
    r = res.results[0]
    assert not r.error, r.error
    assert r.metrics["roc_auc"] > 0.7, (spec.key, r.metrics)
    assert r.oof.shape == (len(y), 2) and np.allclose(r.oof.sum(1), 1, atol=1e-6)
    assert r.pipeline is not None


@pytest.mark.parametrize("spec", REGRESSORS, ids=lambda s: s.key)
def test_every_regressor_runs(spec, heart_df):
    ds = Dataset.from_frame("h", heart_df.copy())
    df, types = ds.base(), ds.base_types()
    feats = [c for c in df.columns if c not in ("thalach", "num")]
    X, y, _, num, cat = prepare_xy(df, "thalach", feats, types, "regression")
    cfg = ev.EvalConfig(task="regression", n_splits=3, n_repeats=1, n_boot=30)
    res = ev.evaluate_models(X, y, [], num, cat, [(spec, {})], cfg)
    r = res.results[0]
    assert not r.error, r.error
    # thalach 噪声大（即便已知诊断结果 R² 上限也只有约 0.25），这里只检查能跑通且指标有限
    assert np.isfinite(r.metrics["r2"]) and r.metrics["r2"] < 1 and r.oof.shape == (len(y),)


@pytest.mark.parametrize("key", ["linreg", "ridge", "rf_r", "gb_r", "xgb_r", "lgbm_r", "svr"])
def test_regression_learns_deterministic_signal(key, heart_df):
    """机制自检：目标是特征的确定函数 + 小噪声时，回归管道必须学得到。"""
    df = heart_df.copy()
    rng = np.random.default_rng(0)
    df["syn"] = 2 * df["age"] + 0.5 * df["chol"] + rng.normal(0, 3, len(df))
    ds = Dataset.from_frame("h", df)
    d, t = ds.base(), ds.base_types()
    feats = [c for c in d.columns if c not in ("syn", "num")]
    X, y, _, num, cat = prepare_xy(d, "syn", feats, t, "regression")
    cfg = ev.EvalConfig(task="regression", n_splits=3, n_repeats=1, n_boot=10)
    r = ev.evaluate_models(X, y, [], num, cat, [(REGISTRY[key], {})], cfg).results[0]
    assert not r.error, r.error
    assert r.metrics["r2"] > 0.85, (key, r.metrics["r2"])


def test_repeated_cv_oof_is_average_and_ci(clf_data):
    X, y, classes, num, cat = clf_data
    cfg = ev.EvalConfig(n_splits=5, n_repeats=2, n_boot=100)
    res = ev.evaluate_models(X, y, classes, num, cat, [(REGISTRY["logreg"], {}), (REGISTRY["rf"], {"n_estimators": 50})], cfg)
    r = res.results[0]
    assert len(r.fold_scores) == 10
    lo, hi = r.ci["roc_auc"]
    assert lo < r.metrics["roc_auc"] < hi + 0.05
    assert "AUC" in res.table().columns and len(res.table()) == 2


def test_holdout_and_class_weight_and_progress(clf_data):
    X, y, classes, num, cat = clf_data
    seen = []
    cfg = ev.EvalConfig(strategy="holdout", class_weight="balanced", n_boot=20)
    res = ev.evaluate_models(X, y, classes, num, cat, [(REGISTRY["logreg"], {})], cfg, progress=lambda a, b, m: seen.append((a, b)))
    r = res.results[0]
    assert len(r.oof_index) == int(round(len(y) * 0.25)) and seen[-1][0] == seen[-1][1]


def test_cancel(clf_data):
    X, y, classes, num, cat = clf_data
    with pytest.raises(ev.Cancelled):
        ev.evaluate_models(X, y, classes, num, cat, [(REGISTRY["rf"], {})], ev.EvalConfig(), cancel=lambda: True)


def test_failure_isolated(clf_data):
    X, y, classes, num, cat = clf_data
    res = ev.evaluate_models(X, y, classes, num, cat, [(REGISTRY["rf"], {"n_estimators": -5}), (REGISTRY["logreg"], {})],
                             ev.EvalConfig(n_splits=3, n_repeats=1, n_boot=10))
    assert res.results[0].error and not res.results[1].error


def test_threshold_tools(clf_data):
    X, y, classes, num, cat = clf_data
    res = ev.evaluate_models(X, y, classes, num, cat, [(REGISTRY["logreg"], {})], ev.EvalConfig(n_splits=3, n_repeats=1, n_boot=10))
    r = res.results[0]
    yy, p = res.y_oof(r), r.oof[:, 1]
    t = ev.threshold_table(yy, p)
    assert t["灵敏度"].is_monotonic_decreasing and t["特异度"].is_monotonic_increasing
    thr = ev.threshold_for_sensitivity(yy, p, 0.95)
    assert ((p >= thr) & (yy == 1)).sum() / (yy == 1).sum() >= 0.95
    assert 0 < ev.youden_threshold(yy, p) < 1


def test_multiclass(heart_df):
    ds = Dataset.from_frame("h", heart_df.copy())
    df, types = ds.base(), ds.base_types()
    feats = [c for c in df.columns if c != "num"]
    X, y, classes, num, cat = prepare_xy(df, "num", feats, types, "classification")
    res = ev.evaluate_models(X, y, classes, num, cat, [(REGISTRY["rf"], {"n_estimators": 60}), (REGISTRY["xgb"], {"n_estimators": 40})],
                             ev.EvalConfig(n_splits=3, n_repeats=1, n_boot=10))
    for r in res.results:
        assert not r.error, r.error
        assert r.oof.shape == (len(y), 5) and "roc_auc" in r.metrics


def test_tune(clf_data):
    X, y, classes, num, cat = clf_data
    best, score, tab = ev.tune_model(REGISTRY["knn"], X, y, num, cat, ev.EvalConfig(n_splits=3), n_iter=4)
    assert "n_neighbors" in best and score > 0.7 and len(tab) >= 4


def test_permutation_and_native(clf_data):
    X, y, classes, num, cat = clf_data
    cfg = ev.EvalConfig(n_splits=3, n_repeats=1, n_boot=10)
    imp = ex.heldout_permutation_importance(X, y, REGISTRY["rf"], {"n_estimators": 60}, num, cat, cfg, n_repeats=5, n_splits=3)
    assert set(imp["变量"]) == set(X.columns)
    assert imp.iloc[0]["重要性(得分下降)"] > 0
    res = ev.evaluate_models(X, y, classes, num, cat, [(REGISTRY["logreg"], {})], cfg)
    nat = ex.native_importance(res.results[0].pipeline)
    assert nat is not None and "系数（标准化后）" in nat.columns


@pytest.mark.parametrize("key", ["logreg", "rf", "xgb", "svm"])
def test_shap_binary(key, clf_data):
    pytest.importorskip("shap")
    X, y, classes, num, cat = clf_data
    cfg = ev.EvalConfig(n_splits=3, n_repeats=1, n_boot=10)
    spec = REGISTRY[key]
    res = ev.evaluate_models(X, y, classes, num, cat, [(spec, {"n_estimators": 40} if key in ("rf", "xgb") else {})], cfg)
    sh = ex.compute_shap(res.results[0].pipeline, spec, X.head(80), "classification")
    assert sh.values.shape == (80, len(sh.names)) and np.isfinite(sh.values).all()
    sv, Z, base = sh.explain(X.iloc[[0]])
    assert sv.shape == (1, len(sh.names))
    assert len(sh.mean_abs()) == len(sh.names)


def test_clustering(heart_ds):
    df, types = heart_ds.base(), heart_ds.base_types()
    feats = ["age", "trestbps", "chol", "thalach", "oldpeak", "sex", "cp"]
    for spec in CLUSTERERS:
        r = cl.run_clustering(df, feats, types, spec, {} if spec.key != "dbscan" else {"eps": 2.5})
        assert len(r.labels) == len(df)
        if spec.key != "dbscan":
            assert r.n_clusters == 3 and -1 <= r.metrics["silhouette"] <= 1
    sweep = cl.k_sweep(r.Xt, 2, 5)
    assert list(sweep["K"]) == [2, 3, 4, 5]
    r = cl.run_clustering(df, feats, types, REGISTRY["kmeans"], {"n_clusters": 2})
    prof = cl.cluster_profile(df, r.labels, feats, types)
    assert {"簇0", "簇1"} <= set(prof.columns)
    ct, met = cl.compare_with_truth(r.labels, df["num"])
    assert ct.shape[0] == 2 and "ARI（调整兰德指数）" in met
    emb, var = cl.embed_2d(r.Xt)
    assert emb.shape == (len(df), 2)
