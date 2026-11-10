"""⑦ 预测：对单个样本做风险预测与解释；批量预测；载入/使用已保存模型。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)

from ...core import explain as ex
from ...core import io as mio
from ...core import persist
from ...core import plots as pl
from ...core.medical import value_labels
from ..widgets.plotwidget import PlotWidget
from ..widgets.tables import DataTable
from ..widgets.tasks import TaskRunner
from .base import Page, group, h1, hint, scroll


class PredictPage(Page):
    key, title = "predict", "预测"

    def __init__(self, state):
        super().__init__(state)
        self._pkg: persist.ModelPackage | None = None
        self._ext_pkg: persist.ModelPackage | None = None  # 从文件载入的模型
        self._inputs: dict[str, QWidget] = {}
        self._shap_cache: dict[str, object] = {}
        lay = self.root_layout()
        lay.addWidget(h1("预测与解释"))
        lay.addWidget(hint("在“建模评估”训练后，可输入一个样本的各项指标，得到风险概率和主要影响因素；也可批量预测整个文件，或载入之前保存的模型。"))

        bar = QHBoxLayout()
        bar.addWidget(QLabel("使用模型："))
        self.src = QComboBox()
        self.src.setMinimumWidth(320)
        self.src.currentIndexChanged.connect(self._source_changed)
        bar.addWidget(self.src)
        b1 = QPushButton("载入模型文件…")
        b1.clicked.connect(self._load)
        self.btn_batch = QPushButton("批量预测文件…")
        self.btn_batch.clicked.connect(self._batch)
        bar.addWidget(b1)
        bar.addWidget(self.btn_batch)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.model_info = hint("")
        lay.addWidget(self.model_info)

        split = QSplitter(Qt.Horizontal)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        self.form.setLabelAlignment(Qt.AlignLeft)
        split.addWidget(scroll(self.form_host, 380))
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.btn_pred = QPushButton("预测")
        self.btn_pred.setObjectName("primary")
        self.btn_pred.clicked.connect(self._predict)
        self.btn_reset = QPushButton("恢复为中位数/众数")
        self.btn_reset.clicked.connect(self._reset_inputs)
        self.btn_ai = QPushButton("交给 AI 解读此样本 →")
        self.btn_ai.clicked.connect(lambda: self.state.goto_page.emit("ai"))
        for b in (self.btn_pred, self.btn_reset, self.btn_ai):
            row.addWidget(b)
        self.run_lbl = QLabel("")  # 任务进度文字（与模型信息分开显示）
        row.addWidget(self.run_lbl)
        row.addStretch(1)
        rl.addLayout(row)
        self.result = hint("")
        rl.addWidget(self.result)
        self.p_risk = PlotWidget(figsize=(6, 1.6), with_report_button=False)
        self.p_risk.setMaximumHeight(150)
        rl.addWidget(self.p_risk)
        self.p_shap = PlotWidget(figsize=(6, 4))
        self.p_shap.add_to_report.connect(lambda p: self.state.report_add_figure(p, "单个样本的主要影响因素", "预测"))
        rl.addWidget(self.p_shap, 1)
        split.addWidget(right)
        split.setSizes([380, 780])
        lay.addWidget(split, 1)
        self.runner = TaskRunner(self, None, self.run_lbl, self.btn_pred)

        state.eval_done.connect(self._refresh_sources)
        state.dataset_changed.connect(self._refresh_sources)
        self._refresh_sources()

    # ---------- 模型来源 ----------
    def _eval_pkg(self, key: str):
        b = self.state.eval
        r = b.res.get(key)
        return persist.make_package(r.name, r.pipeline, b.task, b.target, b.res.X, b.res.num_cols, b.res.cat_cols,
                                    b.res.classes, b.threshold, r.metrics, b.specs[key].key)

    def _refresh_sources(self):
        self.src.blockSignals(True)
        self.src.clear()
        b = self.state.eval
        if b:
            for r in b.res.results:
                if not r.error:
                    self.src.addItem(f"当前评估：{r.name}", ("eval", r.key))
        if self._ext_pkg:
            self.src.addItem(f"已载入文件：{self._ext_pkg.name}（目标 {self._ext_pkg.target}）", ("file", None))
        if b and self.src.count():
            i = self.src.findData(("eval", b.selected))
            self.src.setCurrentIndex(max(0, i))
        self.src.blockSignals(False)
        self._source_changed()

    def _source_changed(self):
        data = self.src.currentData()
        self._pkg = None
        if data:
            kind, key = data
            self._pkg = self._eval_pkg(key) if kind == "eval" else self._ext_pkg
        self._build_form()

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "载入模型文件", "", "MedMiner 模型 (*.joblib)")
        if not path:
            return
        if QMessageBox.question(self, "安全提示", "模型文件在载入时可能执行其中包含的代码。\n只应载入你自己生成或完全信任的文件。\n\n继续载入？") != QMessageBox.Yes:
            return
        try:
            self._ext_pkg = persist.load_package(path)
        except Exception as e:
            self.error("无法载入模型", f"{type(e).__name__}: {e}")
            return
        self._refresh_sources()
        self.src.setCurrentIndex(self.src.findData(("file", None)))

    # ---------- 表单 ----------
    def _build_form(self):
        while self.form.rowCount():
            self.form.removeRow(0)
        self._inputs.clear()
        pkg = self._pkg
        self.btn_pred.setEnabled(pkg is not None)
        self.btn_batch.setEnabled(pkg is not None)
        self.btn_ai.setEnabled(pkg is not None and pkg.task == "classification")
        if pkg is None:
            self.model_info.setText("尚无可用模型：请先在“建模评估”训练，或载入模型文件。")
            self.p_risk.message("")
            self.p_shap.message("")
            return
        ds = self.state.ds
        raw = ds.raw if ds else pd.DataFrame()
        m = ", ".join(f"{k}={v:.3f}" for k, v in list(pkg.metrics.items())[:4])
        self.model_info.setText(f"模型：{pkg.name}　任务：{'分类' if pkg.task == 'classification' else '回归'}　目标：{pkg.target}　"
                                f"训练样本 {pkg.n_train}　{m}")
        for c in pkg.features:
            label = QLabel(ds.label(c) if ds and c in ds.raw.columns else c)
            if c in pkg.num_cols and c in pkg.ranges:
                lo, med, hi = pkg.ranges[c]
                sp = QDoubleSpinBox()
                span = max(hi - lo, 1.0)
                sp.setRange(lo - span, hi + span)
                integer = float(lo).is_integer() and float(hi).is_integer() and float(med).is_integer()
                sp.setDecimals(0 if integer else 2)
                sp.setValue(med)
                sp.setToolTip(f"训练数据范围：{lo:g} ~ {hi:g}，中位数 {med:g}")
                sp._default = med
                self.form.addRow(label, sp)
                self._inputs[c] = sp
            else:
                cb = QComboBox()
                labels = value_labels(raw, c) if len(raw) else {}
                vals = pkg.categories.get(c, [])
                for v in vals:
                    lab = labels.get(v, labels.get(int(v) if isinstance(v, float) and v.is_integer() else v))
                    cb.addItem(f"{v:g}" + (f"  {lab}" if lab else "") if isinstance(v, (int, float)) else str(v), v)
                cb._default = vals[0] if vals else None
                if ds is not None and c in ds.base().columns and ds.base()[c].notna().any():
                    mode = ds.base()[c].mode().iloc[0]
                    j = cb.findData(mode.item() if hasattr(mode, "item") else mode)
                    if j >= 0:
                        cb.setCurrentIndex(j)
                        cb._default = cb.itemData(j)
                self.form.addRow(label, cb)
                self._inputs[c] = cb

    def _reset_inputs(self):
        for w in self._inputs.values():
            if isinstance(w, QDoubleSpinBox):
                w.setValue(w._default)
            else:
                j = w.findData(w._default)
                if j >= 0:
                    w.setCurrentIndex(j)

    def _row(self) -> pd.DataFrame:
        vals = {}
        for c, w in self._inputs.items():
            vals[c] = float(w.value()) if isinstance(w, QDoubleSpinBox) else w.currentData()
        return pd.DataFrame([vals], columns=self._pkg.features)

    # ---------- 预测 ----------
    def _predict(self):
        pkg = self._pkg
        if pkg is None:
            return
        X = self._row()
        warn = []
        for c in pkg.num_cols:
            if c in pkg.ranges:
                lo, _, hi = pkg.ranges[c]
                v = X.iloc[0][c]
                if v < lo or v > hi:
                    warn.append(f"{c}={v:g} 超出训练范围 [{lo:g}, {hi:g}]")
        wtxt = ("　⚠ " + "；".join(warn) + "（外推，预测不可靠）") if warn else ""
        if pkg.task == "regression":
            pred = float(pkg.pipeline.predict(X)[0])
            self.result.setText(f"预测 {pkg.target} = {pred:.3f}{wtxt}")
            self.result.setObjectName("warn" if warn else "ok")
            self.result.style().unpolish(self.result)
            self.result.style().polish(self.result)
            self.p_risk.message(f"预测值 {pred:.3f}")
            self.p_shap.message("回归模型暂不提供单样本解释")
            return
        proba = pkg.pipeline.predict_proba(X)[0]
        if len(pkg.classes) != 2:
            top = int(proba.argmax())
            self.result.setText("预测类别：" + pkg.classes[top] + "（概率 " + f"{proba[top]:.1%}" + "）　各类概率：" +
                                "，".join(f"{c}:{p:.1%}" for c, p in zip(pkg.classes, proba)) + wtxt)
            self.p_risk.message(f"预测类别 {pkg.classes[top]}")
            self.p_shap.message("多分类模型暂不提供单样本解释")
            return
        p, thr = float(proba[1]), pkg.threshold
        high = p >= thr
        self.result.setText(f"预测“{pkg.classes[1]}”的概率 {p:.1%}　阈值 {thr:.2f}　→　{'高风险' if high else '低风险'}　"
                            f"（模型输出，仅供教学研究，不能替代临床判断）{wtxt}")
        self.result.setObjectName("warn" if high or warn else "ok")
        self.result.style().unpolish(self.result)
        self.result.style().polish(self.result)
        self.p_risk.draw_with(pl.plot_single_risk, p, thr)
        self._explain(X, p, thr)

    def _explain(self, X: pd.DataFrame, p: float, thr: float):
        pkg = self._pkg
        b = self.state.eval
        data = self.src.currentData()
        if data is None or data[0] != "eval" or b is None:
            self.p_shap.message("已载入的模型文件不含训练数据，无法计算 SHAP 解释（仅显示概率）")
            self._remember_patient(X, p, thr, None)
            return
        key = data[1]
        spec = b.specs[key]
        Xref = b.res.X

        def job(progress, cancel):
            progress(0, 0, "计算 SHAP 解释…")
            sh = self._shap_cache.get(key)
            if sh is None:
                sh = ex.compute_shap(pkg.pipeline, spec, Xref, "classification", 2, max_rows=150)
            sv, Z, base = sh.explain(X)
            return key, sh, sv, Z, base

        def done(out):
            k, sh, sv, Z, base = out
            self._shap_cache[k] = sh
            names = sh.names
            self.p_shap.draw_with(pl.plot_shap_single, names, sv[0], Z.iloc[0].to_numpy(dtype=float), base, sh.output)
            contrib = pd.DataFrame({"特征": names, "取值": Z.iloc[0].to_numpy(dtype=float), "SHAP": sv[0]})
            contrib = contrib.reindex(contrib["SHAP"].abs().sort_values(ascending=False).index).reset_index(drop=True)
            self._remember_patient(X, p, thr, contrib)

        def fail(msg):
            self.p_shap.message(f"无法计算解释：{msg}")
            self._remember_patient(X, p, thr, None)

        self.runner.start(job, done, fail, busy_text="计算 SHAP 解释…")

    def _remember_patient(self, X, p, thr, contrib):
        self.state.cache["patient"] = dict(values={c: X.iloc[0][c] for c in X.columns}, prob=p, threshold=thr,
                                          contrib=contrib, model=self._pkg.name)

    # ---------- 批量预测 ----------
    def _batch(self):
        pkg = self._pkg
        if pkg is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择要预测的数据文件", "", mio.FILE_FILTER)
        if not path:
            return
        try:
            df, _ = mio.load_table(path)
            out = persist.predict_frame(pkg, df)
        except Exception as e:
            self.error("批量预测失败", f"{type(e).__name__}: {e}")
            return
        res = pd.concat([df.reset_index(drop=True), out], axis=1)
        save, _ = QFileDialog.getSaveFileName(self, "保存预测结果", "predictions.csv", "CSV (*.csv);;Excel (*.xlsx)")
        if not save:
            return
        if save.lower().endswith(".xlsx"):
            res.to_excel(save, index=False)
        else:
            res.to_csv(save, index=False, encoding="utf-8-sig")
        QMessageBox.information(self, "完成", f"已预测 {len(res)} 行，结果保存到：\n{save}")
