import numpy as np
import pandas as pd

from medminer.core import stats
from medminer.core.schema import infer_types


def test_describe_and_normality(heart_df):
    t = infer_types(heart_df)
    d = stats.describe(heart_df, t)
    assert len(d) == heart_df.shape[1] and "缺失%" in d.columns
    assert d.loc[d["变量"] == "ca", "缺失%"].iloc[0] > 0
    n = stats.normality(heart_df, ["age", "chol"])
    assert len(n) == 2


def test_table_one(heart_df):
    df = heart_df.copy()
    df["dx"] = (df["num"] > 0).astype(int)
    t = infer_types(df)
    tab = stats.table_one(df, "dx", t, cols=[c for c in df.columns if c not in ("num", "dx")],
                          group_labels={0: "无病", 1: "患病"})
    assert tab.columns[0] == "变量" and "p 值" in tab.columns
    row = tab[tab["变量"].str.startswith("thalach")].iloc[0]
    assert row["检验方法"] in ("Welch t 检验", "Mann-Whitney U")
    assert row["p 值"] in ("<0.001",) or float(row["p 值"]) < 0.05  # 模拟数据里最大心率与患病强相关
    sex_row = tab[tab["变量"].str.startswith("sex")].iloc[0]
    assert sex_row["检验方法"] in ("卡方检验", "Fisher 精确")
    assert (tab["变量"] == "    1").any() or any(v.strip() == "1" for v in tab["变量"])


def test_logit_or(heart_df):
    df = heart_df.copy()
    df["dx"] = (df["num"] > 0).astype(int)
    t = infer_types(df)
    feats = ["age", "thalach", "sex", "cp"]
    uni = stats.logit_or(df, "dx", feats, t)
    assert {"age", "thalach", "sex=1"} <= set(uni["变量"])
    assert uni.loc[uni["变量"] == "thalach", "OR"].iloc[0] < 1  # 最大心率越高越不容易患病
    multi = stats.logit_or(df, "dx", feats, t, multivariable=True)
    assert (multi["95%CI下限"] < multi["OR"]).all() and (multi["OR"] < multi["95%CI上限"]).all()


def test_target_association_and_corr(heart_df):
    df = heart_df.copy()
    df["dx"] = (df["num"] > 0).astype(int)
    t = infer_types(df)
    assoc = stats.target_association(df, "dx", t)
    assert not assoc.empty and assoc["关联强度"].abs().is_monotonic_decreasing
    c = stats.correlation(df, ["age", "chol", "thalach"], "spearman")
    assert c.shape == (3, 3)
