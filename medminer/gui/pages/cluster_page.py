"""⑥ 聚类分析：患者亚型发现。"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ...core import cluster as cl
from ...core import plots as pl
from ...core.models import CLUSTERERS, REGISTRY
from ...core.schema import BINARY, CATEGORICAL, NUMERIC
from ..state import ClusterBundle
from ..widgets.forms import ColumnPickerRow, ParamForm
from ..widgets.plotwidget import PlotWidget
from ..widgets.tables import DataTable
from ..widgets.tasks import TaskRunner
from .base import Page, group, h1, hint, scroll


class ClusterPage(Page):
    key, title = "cluster", "聚类分析"

    def __init__(self, state):
        super().__init__(state)
        self._params: dict[str, dict] = {}
        self._form_key = None
        lay = self.root_layout()
        lay.addWidget(h1("聚类分析（患者亚型发现）"))
        lay.addWidget(hint("无监督：不使用诊断标签就把样本分组，再看每组的临床特征画像，并可与真实标签对照。"
                           "数值特征会标准化、分类特征独热编码后再聚类，缺失值自动填充。"))
        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll(self._left(), 340))
        split.addWidget(self._right())
        split.setSizes([340, 980])
        lay.addWidget(split, 1)
        self.runner = TaskRunner(self, self.bar, self.status_lbl, self.btn_run)
        state.dataset_changed.connect(self._on_dataset)
        state.data_updated.connect(self._fill_features)
        self._alg_changed()

    def _left(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 6, 0)
        f = QFormLayout()
        self.alg = QComboBox()
        for s in CLUSTERERS:
            self.alg.addItem(s.name, s.key)
        self.alg.currentIndexChanged.connect(self._alg_changed)
        f.addRow("算法", self.alg)
        l.addWidget(group("算法", f))
        self.form = ParamForm()
        self.form.changed.connect(self._save_form)
        l.addWidget(group("参数", self._wrap(self.form)))
        self.feat = ColumnPickerRow(height=230)
        l.addWidget(group("参与聚类的特征（建议不含诊断/目标列）", self._wrap(self.feat)))
        g = QFormLayout()
        self.scale = QComboBox()
        for t, d in (("标准化", "standard"), ("MinMax", "minmax"), ("稳健缩放", "robust")):
            self.scale.addItem(t, d)
        self.embed = QComboBox()
        self.embed.addItem("PCA（快，保持全局结构）", "pca")
        self.embed.addItem("t-SNE（慢，突出局部簇）", "tsne")
        self.truth = QComboBox()
        self.seed = QSpinBox()
        self.seed.setRange(0, 99999)
        self.seed.setValue(42)
        g.addRow("缩放", self.scale)
        g.addRow("降维可视化", self.embed)
        g.addRow("对照标签", self.truth)
        g.addRow("随机种子", self.seed)
        l.addWidget(group("其他", g))
        self.btn_run = QPushButton("运行聚类")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._run)
        self.btn_k = QPushButton("帮我选 K（肘部法 + 轮廓系数）")
        self.btn_k.clicked.connect(self._sweep)
        l.addWidget(self.btn_run)
        l.addWidget(self.btn_k)
        self.bar = QProgressBar()
        l.addWidget(self.bar)
        self.status_lbl = hint("")
        l.addWidget(self.status_lbl)
        l.addStretch(1)
        return w

    @staticmethod
    def _wrap(widget) -> QVBoxLayout:
        v = QVBoxLayout()
        v.addWidget(widget)
        return v

    def _right(self) -> QWidget:
        self.tabs = QTabWidget()
        self.p_scatter = PlotWidget(figsize=(8, 5))
        self.p_scatter.add_to_report.connect(lambda p: self.state.report_add_figure(p, "聚类结果", "聚类分析"))
        self.tabs.addTab(self.p_scatter, "聚类图")
        k = QSplitter(Qt.Vertical)
        self.p_k = PlotWidget(figsize=(8, 3.6))
        self.p_k.add_to_report.connect(lambda p: self.state.report_add_figure(p, "选 K：肘部法与轮廓系数", "聚类分析"))
        self.t_k = DataTable(highlight_nan=False)
        k.addWidget(self.p_k)
        k.addWidget(self.t_k)
        k.setSizes([380, 180])
        self.tabs.addTab(k, "选 K")
        prof = QSplitter(Qt.Vertical)
        self.p_heat = PlotWidget(figsize=(8, 4))
        self.p_heat.add_to_report.connect(lambda p: self.state.report_add_figure(p, "簇画像热图", "聚类分析"))
        pw = QWidget()
        pl_ = QVBoxLayout(pw)
        pl_.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("各簇特征画像（数值=均值及相对总体的 σ 倍数；分类=众数及占比）"))
        row.addStretch(1)
        b = QPushButton("画像表加入报告")
        b.clicked.connect(lambda: self._table_to_report(self.t_prof, "簇画像表"))
        row.addWidget(b)
        pl_.addLayout(row)
        self.t_prof = DataTable(highlight_nan=False)
        pl_.addWidget(self.t_prof, 1)
        prof.addWidget(self.p_heat)
        prof.addWidget(pw)
        prof.setSizes([340, 260])
        self.tabs.addTab(prof, "簇画像")
        tr = QWidget()
        trl = QVBoxLayout(tr)
        self.truth_info = hint("")
        trl.addWidget(self.truth_info)
        self.t_cross = DataTable(highlight_nan=False)
        trl.addWidget(self.t_cross, 1)
        self.tabs.addTab(tr, "与真实标签对照")
        self.p_pca = PlotWidget(figsize=(6, 4))
        self.p_pca.add_to_report.connect(lambda p: self.state.report_add_figure(p, "PCA 方差解释", "聚类分析"))
        self.tabs.addTab(self.p_pca, "PCA 方差")
        return self.tabs

    # ---------- 联动 ----------
    def _alg_changed(self):
        self._save_form()
        key = self.alg.currentData()
        spec = REGISTRY[key]
        self._form_key = key
        self.form.set_specs(spec.params, self._params.get(key, {}))

    def _save_form(self):
        if self._form_key:
            self._params[self._form_key] = self.form.values()

    def _on_dataset(self):
        self._fill_features()
        for p in (self.p_scatter, self.p_k, self.p_heat, self.p_pca):
            p.message("尚无结果，请先运行聚类")
        for t in (self.t_prof, self.t_cross, self.t_k):
            t.set_frame(pd.DataFrame())

    def _fill_features(self):
        ds = self.state.ds
        if ds is None:
            return
        df, t = ds.base(), ds.base_types()
        cols = [c for c in df.columns if t.get(c) in (NUMERIC, BINARY, CATEGORICAL)]
        self.feat.set_columns(cols, [c for c in cols if c != ds.target and df[c].nunique() > 1 and not c.startswith("_")],
                              {c: ds.label(c) for c in cols})
        prev = self.truth.currentData()
        self.truth.blockSignals(True)
        self.truth.clear()
        self.truth.addItem("（不对照）", None)
        for c in cols:
            if t.get(c) in (BINARY, CATEGORICAL):
                self.truth.addItem(ds.label(c), c)
        i = self.truth.findData(prev if prev else ds.target)
        self.truth.setCurrentIndex(max(0, i))
        self.truth.blockSignals(False)

    # ---------- 运行 ----------
    def _inputs(self):
        ds = self.state.ds
        feats = self.feat.checked()
        if len(feats) < 2:
            self.error("特征太少", "请至少选择两个特征。")
            return None
        return ds.base(), ds.base_types(), feats

    def _run(self):
        if not self.need_data() or self.runner.busy():
            return
        inp = self._inputs()
        if inp is None:
            return
        df, types, feats = inp
        self._save_form()
        spec = REGISTRY[self.alg.currentData()]
        params = self._params.get(spec.key, {})
        scale, seed, emb = self.scale.currentData(), self.seed.value(), self.embed.currentData()
        truth_col = self.truth.currentData()

        def job(progress, cancel):
            progress(0, 0, "聚类中…")
            res = cl.run_clustering(df, feats, types, spec, params, scale, seed)
            progress(0, 0, "降维与画像…")
            e2, var = cl.embed_2d(res.Xt, emb, seed)
            prof = cl.cluster_profile(df, res.labels, feats, types)
            ct = met = None
            if truth_col and truth_col in df.columns:
                ct, met = cl.compare_with_truth(res.labels, df[truth_col])
            return res, e2, var, prof, ct, met, cl.pca_summary(res.Xt)

        def done(out):
            res, e2, var, prof, ct, met, pca = out
            if res.n_clusters < 1:
                self.error("没有形成有效的簇", "DBSCAN 等算法可能把所有点都当成噪声，请调大 eps 或减小最小样本数。")
                return
            b = ClusterBundle(res, feats, types, df, prof, ct, met, truth_col)
            self.state.cluster = b
            truth = df[truth_col] if truth_col and truth_col in df.columns else None
            self.p_scatter.draw_with(pl.plot_cluster_scatter, e2, res.labels, var, "PC" if emb == "pca" else "t-SNE", truth)
            self.p_heat.draw_with(pl.plot_cluster_heatmap, df, res.labels, feats, types)
            self.t_prof.set_frame(prof)
            self.p_pca.draw_with(pl.plot_pca_variance, pca)
            if ct is not None:
                self.t_cross.set_frame(ct.reset_index())
                self.truth_info.setText("与真实标签的一致性：" + "；".join(f"{k} = {v:.3f}" for k, v in met.items())
                                        + "（ARI/NMI 越接近 1 越一致，接近 0 表示与随机分组无异）")
            else:
                self.t_cross.set_frame(pd.DataFrame())
                self.truth_info.setText("未选择对照标签。")
            m = res.metrics
            self.toast(f"聚类完成：{res.n_clusters} 个簇" + (f"，轮廓系数 {m['silhouette']:.3f}" if m else "")
                       + (f"，噪声点 {res.n_noise}" if res.n_noise else ""))
            self.state.cluster_done.emit()
            self.tabs.setCurrentIndex(0)

        self.runner.start(job, done, busy_text="聚类中…")

    def _sweep(self):
        if not self.need_data() or self.runner.busy():
            return
        inp = self._inputs()
        if inp is None:
            return
        df, types, feats = inp
        scale, seed = self.scale.currentData(), self.seed.value()

        def job(progress, cancel):
            progress(0, 0, "扫描 K=2…10")
            Xt = cl.prepare_matrix(df, feats, types, scale)
            return cl.k_sweep(Xt, 2, 10, seed)

        def done(sw):
            self.p_k.draw_with(pl.plot_k_sweep, sw)
            self.t_k.set_frame(sw.round(4))
            best = int(sw.loc[sw["轮廓系数"].idxmax(), "K"])
            self.toast(f"轮廓系数最高的 K = {best}（仅供参考，请结合肘部形状与临床可解释性选择）")
            self.tabs.setCurrentIndex(1)

        self.runner.start(job, done, busy_text="计算不同 K…")

    def _table_to_report(self, table: DataTable, title: str):
        df = table.frame()
        if df.empty:
            self.error("没有内容", "请先运行聚类。")
            return
        self.state.report_add_table(df, title, "聚类分析")
