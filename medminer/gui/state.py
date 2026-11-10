"""全局应用状态：各页面通过信号协同，不直接互相引用。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from PySide6.QtCore import QObject, Signal

from ..ai.config import LLMConfig, load_config
from ..core.cluster import ClusterResult
from ..core.dataset import Dataset
from ..core.evaluate import EvalResult
from ..core.models import ModelSpec
from ..core.report import Report


@dataclass
class EvalBundle:
    """一次建模评估的完整上下文，供结果展示 / 预测 / AI / 报告共用。"""
    res: EvalResult
    target: str
    features: list[str]
    task: str
    specs: dict[str, ModelSpec]
    types: dict[str, str]  # 建模所用数据的列类型
    in_fold_prep: bool = True
    threshold: float = 0.5
    selected: str = ""  # 当前选中的模型 key
    importance: pd.DataFrame | None = None
    shap: Any = None  # 当前选中模型的 ShapResult


@dataclass
class ClusterBundle:
    res: ClusterResult
    features: list[str]
    types: dict[str, str]
    df: pd.DataFrame
    profile: pd.DataFrame | None = None
    crosstab: pd.DataFrame | None = None
    truth_metrics: dict | None = None
    truth_col: str | None = None


class AppState(QObject):
    dataset_changed = Signal()  # 载入了新数据集
    data_updated = Signal()  # 步骤/类型/目标列变化
    eval_done = Signal()
    cluster_done = Signal()
    report_changed = Signal()
    config_changed = Signal()
    status = Signal(str)
    goto_page = Signal(str)  # 请求主窗口切换页面

    def __init__(self):
        super().__init__()
        self.ds: Dataset | None = None
        self.eval: EvalBundle | None = None
        self.cluster: ClusterBundle | None = None
        self.report = Report()
        self.llm: LLMConfig = load_config()
        self.cache: dict[str, Any] = {}  # 统计页等产生的、供 AI/报告使用的结果：table1 / or / ...
        self.ai_notes: list[tuple[str, str]] = []  # (标题, 文本)

    # ---- 数据集 ----
    def set_dataset(self, ds: Dataset):
        self.ds = ds
        self.eval = None
        self.cluster = None
        self.cache.clear()
        self.dataset_changed.emit()
        self.status.emit(f"已载入数据集：{ds.name}（{ds.raw.shape[0]} 行 × {ds.raw.shape[1]} 列）")

    def touch(self):
        """步骤/类型/目标列被修改。已有的模型与聚类结果基于旧数据，标记为过期。"""
        if self.ds is not None:
            self.ds.invalidate()
        self.data_updated.emit()

    def has_data(self) -> bool:
        return self.ds is not None

    # ---- 报告 ----
    def report_add_table(self, df: pd.DataFrame, title: str, source: str = ""):
        self.report.add_table(df, title, source)
        self.report_changed.emit()
        self.status.emit(f"已加入报告：{title}")

    def report_add_figure(self, png: bytes, title: str, source: str = ""):
        self.report.add_figure(png, title, source)
        self.report_changed.emit()
        self.status.emit(f"已加入报告：{title or '图表'}")

    def report_add_ai(self, text: str, title: str = "AI 分析"):
        self.report.add_ai(text, title)
        self.report_changed.emit()
        self.status.emit("已将 AI 解读加入报告")
