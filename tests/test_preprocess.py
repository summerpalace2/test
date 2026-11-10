import numpy as np
import pandas as pd

from medminer.core.preprocess import OPS, Ctx, Step, default_params, run_pipeline
from medminer.core.schema import infer_types


def run(df, steps, target="num", **kw):
    ctx = Ctx(types=infer_types(df), target=target)
    return run_pipeline(df, steps, ctx, **kw)


def test_every_op_runs_with_defaults(heart_df):
    for op in OPS:
        out, log = run(heart_df, [Step(op, default_params(op))])
        assert log[0]["status"] in ("ok",), (op, log)
        assert len(out) > 0


def test_fill_missing_and_log(heart_df):
    out, log = run(heart_df, [Step("fill_missing", {"strategy": "mode", "columns": ["ca", "thal"]})])
    assert out[["ca", "thal"]].isna().sum().sum() == 0
    assert "→" in log[0]["msg"]


def test_fill_knn(heart_df):
    out, _ = run(heart_df, [Step("fill_missing", {"strategy": "knn", "columns": ["ca"]})])
    assert out["ca"].isna().sum() == 0


def test_fitted_steps_skipped_for_base(heart_df):
    steps = [Step("binarize", {"column": "num", "threshold": 0, "new_name": "dx"}),
             Step("fill_missing", {"strategy": "median"}),
             Step("scale", {"method": "standard", "columns": ["age"]})]
    full, _ = run(heart_df, steps, include_fitted=True)
    base, log = run(heart_df, steps, include_fitted=False)
    assert abs(full["age"].mean()) < 1e-6
    assert base["age"].mean() > 40  # 未缩放
    assert base["ca"].isna().sum() == 4  # 未填充
    assert [e["status"] for e in log] == ["ok", "skipped", "skipped"]
    assert set(base["dx"].dropna().unique()) <= {0, 1}


def test_outliers_actions(heart_df):
    df = heart_df.copy()
    df.loc[0, "chol"] = 5000
    n0 = len(df)
    rem, _ = run(df, [Step("outliers", {"columns": ["chol"], "method": "iqr", "action": "remove"})])
    assert len(rem) < n0 and rem["chol"].max() < 5000
    nan, _ = run(df, [Step("outliers", {"columns": ["chol"], "method": "zscore", "k": 3, "action": "to_nan"})])
    assert np.isnan(nan.loc[0, "chol"])
    clip, _ = run(df, [Step("outliers", {"columns": ["chol"], "method": "mad", "action": "clip"})])
    assert clip["chol"].max() < 5000
    iso, _ = run(df, [Step("outliers", {"columns": ["chol", "age"], "method": "isolation_forest",
                                         "contamination": 0.05, "action": "remove"})])
    assert len(iso) < n0


def test_encode_onehot_label_and_filter(heart_df):
    out, _ = run(heart_df, [Step("encode", {"columns": ["cp"], "method": "onehot"})])
    assert "cp" not in out.columns and any(c.startswith("cp_") for c in out.columns)
    out, _ = run(heart_df, [Step("encode", {"columns": ["thal"], "method": "label"})])
    assert out["thal"].dropna().max() <= 2
    out, _ = run(heart_df, [Step("filter_rows", {"column": "age", "operator": ">=", "value": "60"})])
    assert (out["age"] >= 60).all()


def test_range_filter_and_map(heart_df):
    out, _ = run(heart_df, [Step("range_filter", {"columns": ["thalach"], "min": 100, "max": 180, "action": "to_nan"})])
    assert out["thalach"].dropna().between(100, 180).all()
    out, _ = run(heart_df, [Step("map_values", {"column": "sex", "mapping": "1:男,0:女"})])
    assert set(out["sex"].unique()) == {"男", "女"}


def test_feature_select_and_error_is_isolated(heart_df):
    df = heart_df.copy()
    df["const"] = 1.0
    out, _ = run(df, [Step("feature_select", {"method": "variance", "threshold": 0.0})])
    assert "const" not in out.columns
    out, log = run(heart_df, [Step("fill_missing", {"strategy": "constant", "constant": "abc", "columns": ["ca"]}),
                              Step("drop_duplicates", {})])
    assert log[-1]["status"] == "ok"  # 前一步出错/异常不应阻止后续
