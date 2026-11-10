import numpy as np
import pandas as pd
import pytest

from medminer.core import io as mio
from medminer.core.dataset import Dataset, load_project, save_project
from medminer.core.medical import HEART_COLUMNS, check_clinical_ranges, is_heart_dataset
from medminer.core.preprocess import Step, default_params
from medminer.core.schema import BINARY, CATEGORICAL, NUMERIC, infer_types


def test_load_csv_xlsx_txt(sample_dir, heart_df):
    for name in ["processed.cleveland.csv", "heart_sample.xlsx", "processed.cleveland.data.txt"]:
        df, info = mio.load_table(sample_dir / name)
        assert df.shape == heart_df.shape, name
        assert list(df.columns) == HEART_COLUMNS
        assert df["ca"].isna().sum() == 4  # '?' 被识别为缺失
        assert pd.api.types.is_numeric_dtype(df["ca"])
    _, info = mio.load_table(sample_dir / "processed.cleveland.data.txt")
    assert info["header"] is False and any("14 列" in n for n in info["notes"])


def test_load_gbk_semicolon_and_whitespace(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes("姓名;年龄;血压\n张三;30;120\n李四;40;130\n王五;50;?\n".encode("gb18030"))
    df, info = mio.load_table(p)
    assert list(df.columns) == ["姓名", "年龄", "血压"] and info["delimiter"] == ";"
    assert df["血压"].isna().sum() == 1
    q = tmp_path / "b.txt"
    q.write_text("a b c\n1 2 3\n4 5 6\n")
    df2, _ = mio.load_table(q)
    assert df2.shape == (2, 3)


def test_load_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"data": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]}')
    df, _ = mio.load_table(p)
    assert list(df.columns) == ["a", "b"]


def test_infer_types(heart_df):
    t = infer_types(heart_df)
    for c in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        assert t[c] == NUMERIC, c
    for c in ["sex", "fbs", "exang"]:
        assert t[c] == BINARY, c
    for c in ["cp", "restecg", "slope", "ca", "thal", "num"]:
        assert t[c] == CATEGORICAL, c
    assert is_heart_dataset(heart_df)


def test_dataset_defaults(heart_ds):
    assert heart_ds.target == "num"
    assert "胸痛" in heart_ds.descriptions["cp"]
    assert "（" in heart_ds.label("age")


def test_clinical_range_check(heart_df):
    df = heart_df.copy()
    df.loc[0, "age"] = 200
    out = check_clinical_ranges(df)
    assert out.loc[out["变量"] == "age", "超范围个数"].iloc[0] >= 1


def test_project_roundtrip(tmp_path, heart_ds):
    heart_ds.add_step(Step("drop_duplicates", {}))
    heart_ds.add_step(Step("binarize", {"column": "num", "threshold": 0, "new_name": "y"}))
    save_project(heart_ds, tmp_path / "p.medproj")
    ds2 = load_project(tmp_path / "p.medproj")
    assert ds2.target == "num" and len(ds2.steps) == 2
    assert ds2.current().shape == heart_ds.current().shape
    assert "y" in ds2.current().columns
