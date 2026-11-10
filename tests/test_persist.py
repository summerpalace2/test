import joblib
import numpy as np
import pandas as pd
import pytest

from medminer.core import evaluate as ev
from medminer.core import persist as ps
from medminer.core.dataset import Dataset
from medminer.core.features import prepare_xy
from medminer.core.models import REGISTRY
from medminer.core.preprocess import Step


@pytest.fixture(scope="module")
def setup(heart_df):
    ds = Dataset.from_frame("h", heart_df.copy())
    ds.add_step(Step("binarize", {"column": "num", "threshold": 0, "new_name": "dx"}))
    return ds


def test_leakage_detects_source_column_of_binarized_target(setup):
    df, t = setup.base(), setup.base_types()
    feats = [c for c in df.columns if c != "dx"]  # 故意把 num 留在特征里
    X, y, classes, num, cat = prepare_xy(df, "dx", feats, t, "classification")
    sus = dict(ps.leakage_suspects(X, y, t, "classification"))
    assert "num" in sus and "纯度" in sus["num"]
    assert set(sus) == {"num"}  # 正常特征不会被误报


def test_leakage_detects_column_derived_from_target(setup):
    """反方向：目标是原始多分类 num，特征里留着由它二值化得到的 dx（dx 被目标决定）。"""
    df, t = setup.base(), setup.base_types()
    feats = [c for c in df.columns if c != "num"]
    X, y, classes, num, cat = prepare_xy(df, "num", feats, t, "classification")
    sus = dict(ps.leakage_suspects(X, y, t, "classification"))
    assert "dx" in sus and "派生" in sus["dx"]
    assert set(sus) == {"dx"}  # sex/cp 等正常特征不应误报


def test_leakage_numeric_and_regression(heart_df):
    rng = np.random.default_rng(0)
    df = heart_df.copy()
    df["bp_copy"] = df["trestbps"] * 1.0 + rng.normal(0, 0.01, len(df))
    d = Dataset.from_frame("h", df)
    t = d.base_types()
    X, y, _, _, _ = prepare_xy(d.base(), "trestbps", [c for c in df.columns if c not in ("trestbps",)], t, "regression")
    assert "bp_copy" in dict(ps.leakage_suspects(X, y, t, "regression"))
    df["dx"] = (df["num"] > 0).astype(int)
    df["score"] = df["dx"] * 10 + rng.normal(0, 0.1, len(df))  # 几乎完美区分
    d = Dataset.from_frame("h", df)
    X, y, _, _, _ = prepare_xy(d.base(), "dx", [c for c in df.columns if c != "dx"], d.base_types(), "classification")
    assert "score" in dict(ps.leakage_suspects(X, y, d.base_types(), "classification"))


def test_leakage_inflated_auc_is_real(setup):
    """泄漏的真实危害：把 num 留在特征里时 AUC 会虚高到 ~1，这正是需要检测的原因。"""
    df, t = setup.base(), setup.base_types()
    cfg = ev.EvalConfig(n_splits=3, n_repeats=1, n_boot=10)

    def auc(feats):
        X, y, cl, n, c = prepare_xy(df, "dx", feats, t, "classification")
        return ev.evaluate_models(X, y, cl, n, c, [(REGISTRY["logreg"], {})], cfg).results[0].metrics["roc_auc"]
    clean = [c for c in df.columns if c not in ("dx", "num")]
    assert auc(clean + ["num"]) > 0.995 > auc(clean) + 0.01


def test_package_roundtrip_and_batch_predict(setup, tmp_path):
    df, t = setup.base(), setup.base_types()
    feats = [c for c in df.columns if c not in ("dx", "num")]
    X, y, classes, num, cat = prepare_xy(df, "dx", feats, t, "classification")
    res = ev.evaluate_models(X, y, classes, num, cat, [(REGISTRY["logreg"], {})], ev.EvalConfig(n_splits=3, n_repeats=1, n_boot=10))
    r = res.results[0]
    pkg = ps.make_package("逻辑回归", r.pipeline, "classification", "dx", X, num, cat, classes, 0.4, r.metrics, "logreg")
    assert pkg.ranges["age"][0] < pkg.ranges["age"][1] < pkg.ranges["age"][2] and pkg.categories["sex"] == [0, 1]
    p = tmp_path / "m.joblib"
    ps.save_package(pkg, p)
    pkg2 = ps.load_package(p)
    out = ps.predict_frame(pkg2, df.head(10))
    assert list(out.columns) == ["阳性概率", "预测类别"] and len(out) == 10
    np.testing.assert_allclose(out["阳性概率"], r.pipeline.predict_proba(X.head(10))[:, 1])
    assert set(out["预测类别"]) <= {"0", "1"}
    with pytest.raises(ValueError, match="缺少"):
        ps.predict_frame(pkg2, df[["age"]])
    joblib.dump({"x": 1}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError, match="不是 MedMiner"):
        ps.load_package(tmp_path / "bad.joblib")
