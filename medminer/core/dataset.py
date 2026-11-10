"""数据集容器：原始数据 + 列类型 + 目标列 + 预处理步骤；以及项目文件（.medproj）读写。"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .medical import default_descriptions
from .preprocess import Ctx, Step, run_pipeline
from .schema import IGNORE, infer_column_type, infer_types


@dataclass
class Dataset:
    name: str
    raw: pd.DataFrame
    types: dict[str, str] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    target: str | None = None
    steps: list[Step] = field(default_factory=list)
    source: dict = field(default_factory=dict)
    _cur: pd.DataFrame | None = field(default=None, repr=False)
    _base: pd.DataFrame | None = field(default=None, repr=False)
    _log: list = field(default_factory=list, repr=False)

    # ---- 构造 ----
    @staticmethod
    def from_frame(name: str, df: pd.DataFrame, source: dict | None = None) -> "Dataset":
        ds = Dataset(name=name, raw=df.reset_index(drop=True), source=source or {})
        ds.types = infer_types(ds.raw)
        ds.descriptions = default_descriptions(ds.raw)
        if "num" in ds.raw.columns and ds.descriptions:  # 心脏病数据默认目标
            ds.target = "num"
        return ds

    # ---- 派生数据 ----
    def invalidate(self):
        self._cur = self._base = None

    def _ctx(self) -> Ctx:
        return Ctx(types=dict(self.types), target=self.target)

    def current(self) -> pd.DataFrame:
        """应用全部步骤后的数据。"""
        if self._cur is None:
            self._cur, self._log = run_pipeline(self.raw, self.steps, self._ctx(), include_fitted=True)
        return self._cur

    def base(self) -> pd.DataFrame:
        """只应用 fit-free 步骤后的数据，供建模时在折内再做拟合类预处理。"""
        if self._base is None:
            self._base, _ = run_pipeline(self.raw, self.steps, self._ctx(), include_fitted=False)
        return self._base

    def log(self) -> list[dict]:
        self.current()
        return self._log

    def current_types(self) -> dict[str, str]:
        return self.types_for(self.current())

    def base_types(self) -> dict[str, str]:
        return self.types_for(self.base())

    def types_for(self, df: pd.DataFrame) -> dict[str, str]:
        out = {}
        for c in df.columns:
            out[c] = self.types.get(c) or infer_column_type(df[c])
        return out

    def has_fitted_steps(self) -> bool:
        from .preprocess import OPS
        return any(OPS[s.op].fitted and s.enabled for s in self.steps if s.op in OPS)

    # ---- 步骤编辑 ----
    def add_step(self, step: Step):
        self.steps.append(step)
        self.invalidate()

    def remove_step(self, i: int):
        del self.steps[i]
        self.invalidate()

    def move_step(self, i: int, j: int):
        if 0 <= j < len(self.steps):
            self.steps.insert(j, self.steps.pop(i))
            self.invalidate()

    def set_type(self, col: str, t: str):
        self.types[col] = t
        self.invalidate()

    def set_target(self, col: str | None):
        self.target = col
        self.invalidate()

    def usable_columns(self, df: pd.DataFrame | None = None) -> list[str]:
        df = self.current() if df is None else df
        types = self.types_for(df)
        return [c for c in df.columns if types.get(c) != IGNORE]

    def label(self, col: str) -> str:
        """界面显示名：有中文说明则 “col（中文名）”。"""
        d = self.descriptions.get(col)
        if d and "：" in d:
            return f"{col}（{d.split('：', 1)[0]}）"
        return col


# ---------------- 项目文件 ----------------
def save_project(ds: Dataset, path: str | Path) -> None:
    meta = {"version": 1, "name": ds.name, "types": ds.types, "descriptions": ds.descriptions,
            "target": ds.target, "steps": [s.to_dict() for s in ds.steps], "source": ds.source}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2, default=str))
        z.writestr("data.csv", ds.raw.to_csv(index=False))


def load_project(path: str | Path) -> Dataset:
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json").decode("utf-8"))
        raw = pd.read_csv(io.BytesIO(z.read("data.csv")))
    ds = Dataset(name=meta["name"], raw=raw, types=meta["types"], descriptions=meta.get("descriptions", {}),
                 target=meta.get("target"), steps=[Step.from_dict(s) for s in meta.get("steps", [])],
                 source=meta.get("source", {}))
    return ds
