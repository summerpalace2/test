"""⑤ 建模评估：分类/回归多模型对比、交叉验证、ROC/阈值、特征重要性/SHAP、调参、保存模型。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
                               QSlider, QSpinBox, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ...core import evaluate as ev
from ...core import explain as ex
from ...core import persist
from ...core import plots as pl
from ...core.features import prepare_xy
from ...core.medical import binary_confusion_metrics
from ...core.models import REGISTRY, models_for
from ...core.schema import BINARY, CATEGORICAL, NUMERIC
from ..state import EvalBundle
from ..widgets.forms import ColumnPickerRow, ParamForm
from ..widgets.plotwidget import PlotWidget
from ..widgets.tables import DataTable
from ..widgets.tasks import TaskRunner
from .base import Page, group, h1, hint, scroll

DEFAULT_ON = {"classification": ["logreg", "rf", "xgb"], "regression": ["ridge", "rf_r", "xgb_r"]}


class ModelPage(Page):
    key, title = "model", "建模评估"

    def __init__(self, state):
        super().__init__(state)
        self._params: dict[str, dict] = {}
        self._form_key: str | None = None
        self._perm: dict[str, pd.DataFrame] = {}
        self._shap: dict[str, object] = {}
        lay = self.root_layout()
        lay.addWidget(h1("建模评估"))
        lay.addWidget(hint("默认使用“重复分层 K 折交叉验证”，预处理（填充/缩放/独热）在每一折内部拟合，避免数据泄漏。"
                           "小样本下请以 均值±标准差 与 95% 置信区间为准。"))
        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll(self._build_left(), 372))
        split.addWidget(self._build_right())
        split.setSizes([372, 960])
        lay.addWidget(split, 1)
        self.runner = TaskRunner(self, self.bar, self.status_lbl, self.btn_run, self.btn_cancel)
        state.dataset_changed.connect(self._on_dataset)
        state.data_updated.connect(self._on_data_updated)
        self._task_changed()

    # ====================== 左侧配置 ======================
    def _build_left(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 6, 0)

        f = QFormLayout()
        self.task = QComboBox()
        self.task.addItem("分类（预测类别，如是否患病）", "classification")
        self.task.addItem("回归（预测连续值，如血压）", "regression")
        self.task.currentIndexChanged.connect(self._task_changed)
        self.target = QComboBox()
        self.target.currentIndexChanged.connect(self._target_changed)
        f.addRow("任务", self.task)
        f.addRow("目标列", self.target)
        self.feat = ColumnPickerRow(height=170)
        f.addRow("特征列", self.feat)
        self.infold = QCheckBox("在交叉验证内部拟合预处理（推荐）")
        self.infold.setChecked(True)
        self.infold.setToolTip("勾选：使用预处理页中“清洗类”步骤的结果，缺失填充/缩放/独热在每折训练集内拟合。\n"
                               "不勾选：直接使用预处理页全部步骤的结果（拟合类步骤用了全量数据，会有轻微泄漏）。")
        f.addRow(self.infold)
        self.leak = hint("", warn=True)
        self.leak.setVisible(False)
        f.addRow(self.leak)
        l.addWidget(group("任务与数据", f))

        v = QVBoxLayout()
        self.algs = QListWidget()
        self.algs.setMaximumHeight(190)
        self.algs.currentItemChanged.connect(self._alg_selected)
        v.addWidget(self.algs)
        self.alg_help = hint("")
        v.addWidget(self.alg_help)
        self.form = ParamForm()
        self.form.changed.connect(self._save_form)
        v.addWidget(self.form)
        self.btn_tune = QPushButton("自动调参（当前高亮的算法）…")
        self.btn_tune.clicked.connect(self._tune)
        v.addWidget(self.btn_tune)
        l.addWidget(group("算法（勾选参与对比；点击名称可改参数）", v))

        g = QFormLayout()
        self.strategy = QComboBox()
        self.strategy.addItem("重复分层 K 折交叉验证（推荐）", "cv")
        self.strategy.addItem("留出法（单次划分）", "holdout")
        self.strategy.currentIndexChanged.connect(self._strategy_changed)
        self.k = QSpinBox()
        self.k.setRange(2, 20)
        self.k.setValue(5)
        self.rep = QSpinBox()
        self.rep.setRange(1, 20)
        self.rep.setValue(3)
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.1, 0.5)
        self.test_size.setSingleStep(0.05)
        self.test_size.setValue(0.25)
        self.seed = QSpinBox()
        self.seed.setRange(0, 99999)
        self.seed.setValue(42)
        self.cw = QComboBox()
        self.cw.addItem("无", None)
        self.cw.addItem("balanced（类别不平衡时推荐）", "balanced")
        self.impute = QComboBox()
        self.impute.addItems(["median", "mean"])
        self.scale = QComboBox()
        for t, d in (("按算法需要自动缩放", "auto"), ("全部标准化", "standard"), ("全部 MinMax", "minmax"),
                     ("全部稳健缩放", "robust"), ("不缩放", "none")):
            self.scale.addItem(t, d)
        g.addRow("验证方式", self.strategy)
        g.addRow("折数 K", self.k)
        g.addRow("重复次数", self.rep)
        g.addRow("测试集比例", self.test_size)
        g.addRow("随机种子", self.seed)
        g.addRow("类别权重", self.cw)
        g.addRow("数值缺失填充", self.impute)
        g.addRow("缩放", self.scale)
        l.addWidget(group("验证与预处理", g))

        self.btn_run = QPushButton("开始训练与评估")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._train)
        self.btn_cancel = QPushButton("取消")
        row = QHBoxLayout()
        row.addWidget(self.btn_run, 1)
        row.addWidget(self.btn_cancel)
        l.addLayout(row)
        self.bar = QProgressBar()
        l.addWidget(self.bar)
        self.status_lbl = hint("")
        l.addWidget(self.status_lbl)
        l.addStretch(1)
        self._strategy_changed()
        return w

    # ====================== 右侧结果 ======================
    def _build_right(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(QLabel("当前模型："))
        self.cur = QComboBox()
        self.cur.setMinimumWidth(200)
        self.cur.currentIndexChanged.connect(self._cur_changed)
        top.addWidget(self.cur)
        self.stale = hint("数据或预处理已修改，当前结果基于旧数据，请重新训练。", warn=True)
        self.stale.setVisible(False)
        top.addWidget(self.stale, 1)
        l.addLayout(top)

        self.tabs = QTabWidget()
        # 1 对比
        t1 = QSplitter(Qt.Vertical)
        up = QWidget()
        ul = QVBoxLayout(up)
        ul.setContentsMargins(0, 0, 0, 0)
        self.summary = QTextBrowser()
        self.summary.setMaximumHeight(70)
        ul.addWidget(self.summary)
        self.t_res = DataTable(highlight_nan=False)
        ul.addWidget(self.t_res, 1)
        t1.addWidget(up)
        cmpw = QWidget()
        cl = QVBoxLayout(cmpw)
        cl.setContentsMargins(0, 0, 0, 0)
        mr = QHBoxLayout()
        mr.addWidget(QLabel("对比指标："))
        self.metric_box = QComboBox()
        self.metric_box.currentIndexChanged.connect(self._draw_compare)
        mr.addWidget(self.metric_box)
        mr.addStretch(1)
        cl.addLayout(mr)
        self.p_cmp = PlotWidget(figsize=(7, 3.4))
        self.p_cmp.add_to_report.connect(lambda p: self.state.report_add_figure(p, "模型对比", "建模评估"))
        cl.addWidget(self.p_cmp, 1)
        t1.addWidget(cmpw)
        t1.setSizes([260, 320])
        self.tabs.addTab(t1, "模型对比")
        # 2 曲线
        self.curve_tabs = QTabWidget()
        self.p_roc, self.p_pr, self.p_cal, self.p_dca = (PlotWidget(figsize=(6, 5)) for _ in range(4))
        for p, n in ((self.p_roc, "ROC"), (self.p_pr, "PR"), (self.p_cal, "校准曲线"), (self.p_dca, "决策曲线(DCA)")):
            self.curve_tabs.addTab(p, n)
            p.add_to_report.connect(lambda png, n=n: self.state.report_add_figure(png, n, "建模评估"))
        self.tabs.addTab(self.curve_tabs, "ROC / 校准 / DCA")
        # 3 阈值
        t3 = QWidget()
        t3l = QVBoxLayout(t3)
        sr = QHBoxLayout()
        sr.addWidget(QLabel("决策阈值："))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(5, 95)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self._thr_changed)
        sr.addWidget(self.slider, 1)
        self.thr_lbl = QLabel("0.50")
        self.thr_lbl.setMinimumWidth(40)
        sr.addWidget(self.thr_lbl)
        b1 = QPushButton("约登指数最优")
        b2 = QPushButton("灵敏度 ≥ 90%")
        b1.clicked.connect(lambda: self._auto_thr("youden"))
        b2.clicked.connect(lambda: self._auto_thr("sens"))
        sr.addWidget(b1)
        sr.addWidget(b2)
        t3l.addLayout(sr)
        self.thr_info = hint("")
        t3l.addWidget(self.thr_info)
        pr = QHBoxLayout()
        self.p_cm = PlotWidget(figsize=(4.5, 4))
        self.p_thr = PlotWidget(figsize=(5.5, 4))
        self.p_cm.add_to_report.connect(lambda p: self.state.report_add_figure(p, f"混淆矩阵（阈值 {self.slider.value() / 100:.2f}）", "建模评估"))
        self.p_thr.add_to_report.connect(lambda p: self.state.report_add_figure(p, "阈值-指标曲线", "建模评估"))
        pr.addWidget(self.p_cm)
        pr.addWidget(self.p_thr)
        t3l.addLayout(pr, 1)
        self.tabs.addTab(t3, "混淆矩阵与阈值")
        # 4 重要性
        t4 = QWidget()
        t4l = QVBoxLayout(t4)
        ir = QHBoxLayout()
        self.imp_method = QComboBox()
        self.imp_method.addItem("置换重要性（折外，推荐）", "perm")
        self.imp_method.addItem("模型自带（系数/重要度）", "native")
        self.imp_method.addItem("SHAP 汇总图（二分类/回归）", "shap")
        self.btn_imp = QPushButton("计算")
        self.btn_imp.setObjectName("primary")
        self.btn_imp.clicked.connect(self._importance)
        ir.addWidget(self.imp_method)
        ir.addWidget(self.btn_imp)
        ir.addStretch(1)
        t4l.addLayout(ir)
        isp = QSplitter(Qt.Horizontal)
        self.p_imp = PlotWidget(figsize=(6, 5))
        self.p_imp.add_to_report.connect(lambda p: self.state.report_add_figure(p, f"特征重要性（{self.cur.currentText()}）", "建模评估"))
        isp.addWidget(self.p_imp)
        self.t_imp = DataTable(highlight_nan=False)
        isp.addWidget(self.t_imp)
        isp.setSizes([620, 300])
        t4l.addWidget(isp, 1)
        self.tabs.addTab(t4, "特征重要性 / SHAP")
        # 5 回归诊断
        t5 = QWidget()
        t5l = QHBoxLayout(t5)
        self.p_pv = PlotWidget(figsize=(4.5, 4.5))
        self.p_resid = PlotWidget(figsize=(6, 4))
        self.p_pv.add_to_report.connect(lambda p: self.state.report_add_figure(p, "预测值-真实值", "建模评估"))
        self.p_resid.add_to_report.connect(lambda p: self.state.report_add_figure(p, "残差分析", "建模评估"))
        t5l.addWidget(self.p_pv)
        t5l.addWidget(self.p_resid)
        self.tabs.addTab(t5, "回归诊断")
        # 6 明细
        t6 = QSplitter(Qt.Vertical)
        self.t_fold = DataTable(highlight_nan=False)
        self.params_view = QTextBrowser()
        self.params_view.setMaximumHeight(120)
        t6.addWidget(self.t_fold)
        t6.addWidget(self.params_view)
        self.tabs.addTab(t6, "折间明细 / 参数")
        l.addWidget(self.tabs, 1)

        bot = QHBoxLayout()
        for text, fn in (("加入报告（结果表+主要图）", self._to_report), ("保存当前模型…", self._save_model),
                         ("用当前模型做预测 →", lambda: self.state.goto_page.emit("predict")),
                         ("交给 AI 解读 →", lambda: self.state.goto_page.emit("ai"))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            bot.addWidget(b)
        bot.addStretch(1)
        l.addLayout(bot)
        return w

    # ====================== 配置联动 ======================
    def _task(self) -> str:
        return self.task.currentData()

    def _strategy_changed(self):
        cv = self.strategy.currentData() == "cv"
        self.k.setEnabled(cv)
        self.rep.setEnabled(cv)
        self.test_size.setEnabled(not cv)

    def _task_changed(self):
        task = self._task()
        self.algs.blockSignals(True)
        self.algs.clear()
        for spec in models_for(task):
            it = QListWidgetItem(spec.name)
            it.setData(Qt.UserRole, spec.key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if spec.key in DEFAULT_ON[task] else Qt.Unchecked)
            self.algs.addItem(it)
        self.algs.blockSignals(False)
        self._form_key = None
        self.algs.setCurrentRow(0)
        self.cw.setEnabled(task == "classification")
        self.strategy.setItemText(0, "重复分层 K 折交叉验证（推荐）" if task == "classification" else "重复 K 折交叉验证（推荐）")
        self._fill_target()

    def _alg_selected(self, cur, prev):
        self._save_form()
        if cur is None:
            return
        key = cur.data(Qt.UserRole)
        spec = REGISTRY[key]
        self._form_key = key
        self.form.set_specs(spec.params, self._params.get(key, {}))
        extra = []
        if spec.needs_scaling:
            extra.append("对特征尺度敏感，会自动标准化")
        if spec.tree_based:
            extra.append("树模型，对缺失/尺度不敏感")
        if spec.linear:
            extra.append("线性模型，系数可解释")
        self.alg_help.setText("；".join(extra))
        self.btn_tune.setEnabled(bool(spec.grid()))

    def _save_form(self):
        if self._form_key:
            self._params[self._form_key] = self.form.values()

    def _on_dataset(self):
        self._perm.clear()
        self._shap.clear()
        self._clear_results()
        self._fill_target()

    def _on_data_updated(self):
        self._fill_target(keep=True)
        if self.state.eval is not None:
            self.stale.setVisible(True)

    def _fill_target(self, keep: bool = False):
        ds = self.state.ds
        if ds is None:
            return
        df, t = ds.current(), ds.current_types()
        # 数据页改了目标列 => 跟随；否则保留本页自己选择的目标
        ds_target_changed = ds.target != getattr(self, "_seen_ds_target", object())
        self._seen_ds_target = ds.target
        prev = self.target.currentData() if keep and not ds_target_changed else None
        self.target.blockSignals(True)
        self.target.clear()
        task = self._task()
        for c in df.columns:
            ty = t.get(c)
            ok = ty in (BINARY, CATEGORICAL) if task == "classification" else ty == NUMERIC
            if ok:
                self.target.addItem(ds.label(c), c)
        want = prev if prev else ds.target
        i = self.target.findData(want)
        self.target.setCurrentIndex(i if i >= 0 else 0)
        self.target.blockSignals(False)
        self._target_changed()

    def _target_changed(self):
        ds = self.state.ds
        if ds is None:
            return
        tgt = self.target.currentData()
        base = ds.base() if self.infold.isChecked() else ds.current()
        types = ds.base_types() if self.infold.isChecked() else ds.current_types()
        cols = [c for c in base.columns if types.get(c) in (NUMERIC, BINARY, CATEGORICAL) and c != tgt]
        checked = cols
        # 自动排除疑似泄漏列（如目标由 num 二值化而来时，num 本身）
        self._leak_cols = []
        if tgt and tgt in base.columns and cols:
            try:
                X, y, *_ = prepare_xy(base, tgt, cols, types, self._task())
                self._leak_cols = persist.leakage_suspects(X, y, types, self._task())
            except Exception:
                self._leak_cols = []
        drop = {c for c, _ in self._leak_cols}
        checked = [c for c in cols if c not in drop]
        self.feat.set_columns(cols, checked, {c: ds.label(c) for c in cols})
        if drop:
            self.leak.setText("已自动取消勾选疑似数据泄漏的列：" + "；".join(f"{c}（{r}）" for c, r in self._leak_cols))
        self.leak.setVisible(bool(drop))

    # ====================== 训练 ======================
    def _selected_specs(self):
        self._save_form()
        out = []
        for i in range(self.algs.count()):
            it = self.algs.item(i)
            if it.checkState() == Qt.Checked:
                k = it.data(Qt.UserRole)
                out.append((REGISTRY[k], self._params.get(k, {})))
        return out

    def _config(self) -> ev.EvalConfig:
        return ev.EvalConfig(task=self._task(), strategy=self.strategy.currentData(), n_splits=self.k.value(),
                             n_repeats=self.rep.value(), test_size=self.test_size.value(), seed=self.seed.value(),
                             class_weight=self.cw.currentData() if self._task() == "classification" else None,
                             impute_num=self.impute.currentText(), scale=self.scale.currentData())

    def _prepare(self):
        """校验并准备 (X, y, classes, num, cat, types, tgt)；失败时弹窗并返回 None。"""
        ds = self.state.ds
        tgt = self.target.currentData()
        feats = self.feat.checked()
        task = self._task()
        if not tgt:
            self.error("未选择目标列", "请选择目标列。")
            return None
        if not feats:
            self.error("未选择特征", "请至少勾选一个特征列。")
            return None
        if not self._selected_specs():
            self.error("未选择算法", "请至少勾选一个算法。")
            return None
        use_base = self.infold.isChecked()
        df, types = (ds.base(), ds.base_types()) if use_base else (ds.current(), ds.current_types())
        if tgt not in df.columns:
            self.error("目标列已不存在", f"“{tgt}”在预处理后的数据里不存在，请检查预处理步骤。")
            return None
        X, y, classes, num, cat = prepare_xy(df, tgt, feats, types, task)
        if len(y) < 20:
            self.error("样本太少", f"有效样本仅 {len(y)} 个，无法可靠建模。")
            return None
        if task == "classification":
            cnt = np.bincount(y)
            if len(cnt) < 2:
                self.error("目标只有一个类别", "分类任务需要至少两个类别。")
                return None
            if cnt.min() < 2:
                small = [classes[i] for i in np.where(cnt < 2)[0]]
                self.error("类别样本过少", f"类别 {small} 只有 1 个样本，无法分层验证。请合并类别（预处理→取值映射/二值化）或排除。")
                return None
        if not num and not cat:
            self.error("没有可用特征", "所选特征的类型都不是数值/分类。请在“数据管理”检查列类型。")
            return None
        sus = persist.leakage_suspects(X, y, types, task)
        if sus:
            msg = "下列特征几乎能直接推出目标，结果会虚高：\n\n" + "\n".join(f"• {c}：{r}" for c, r in sus)
            box = QMessageBox(self)
            box.setWindowTitle("疑似数据泄漏")
            box.setIcon(QMessageBox.Warning)
            box.setText(msg)
            b_drop = box.addButton("排除这些列并继续（推荐）", QMessageBox.AcceptRole)
            b_keep = box.addButton("仍然使用", QMessageBox.DestructiveRole)
            box.addButton("取消", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is b_drop:
                drop = {c for c, _ in sus}
                X = X.drop(columns=list(drop))
                num = [c for c in num if c not in drop]
                cat = [c for c in cat if c not in drop]
                self.feat.set_checked([c for c in feats if c not in drop])
                if not num and not cat:
                    self.error("没有可用特征", "排除后没有剩余特征。")
                    return None
            elif box.clickedButton() is not b_keep:
                return None
        return X, y, classes, num, cat, types, tgt

    def _train(self):
        if not self.need_data() or self.runner.busy():
            return
        prep = self._prepare()
        if prep is None:
            return
        X, y, classes, num, cat, types, tgt = prep
        specs, cfg, task = self._selected_specs(), self._config(), self._task()
        in_fold = self.infold.isChecked()
        self.stale.setVisible(False)

        def job(progress, cancel):
            return ev.evaluate_models(X, y, classes, num, cat, specs, cfg, progress, cancel)

        def done(res):
            self._perm.clear()
            self._shap.clear()
            bundle = EvalBundle(res, tgt, list(X.columns), task, {s.key: s for s, _ in specs}, types, in_fold)
            ok = [r for r in res.results if not r.error]
            if not ok:
                self.error("全部模型训练失败", "\n".join(f"{r.name}: {r.error}" for r in res.results))
                return
            best = max(ok, key=lambda r: r.metrics.get(ev.default_metric(task), -1e9))
            bundle.selected = best.key
            self.state.eval = bundle
            self._show_results()
            self.state.eval_done.emit()
            fails = [r for r in res.results if r.error]
            self.toast(f"训练完成：最佳 {best.name}（{ev.METRIC_ZH[ev.default_metric(task)]} {best.metrics[ev.default_metric(task)]:.3f}）"
                       + (f"；{len(fails)} 个模型失败，见结果表" if fails else ""))

        self.runner.start(job, done, busy_text="准备训练…")

    # ====================== 展示 ======================
    def _clear_results(self):
        self.t_res.set_frame(pd.DataFrame())
        self.summary.setHtml("")
        self.cur.blockSignals(True)
        self.cur.clear()
        self.cur.blockSignals(False)
        for p in (self.p_cmp, self.p_roc, self.p_pr, self.p_cal, self.p_dca, self.p_cm, self.p_thr, self.p_imp, self.p_pv, self.p_resid):
            p.message("尚无结果，请先训练模型")
        self.t_imp.set_frame(pd.DataFrame())
        self.t_fold.set_frame(pd.DataFrame())
        self.stale.setVisible(False)

    def _bundle(self) -> EvalBundle | None:
        return self.state.eval

    def _cur_result(self):
        b = self._bundle()
        return b.res.get(self.cur.currentData()) if b and self.cur.currentData() else None

    def _show_results(self):
        b = self._bundle()
        res = b.res
        task = b.task
        clf = task == "classification"
        n_cls = len(res.classes)
        binary = clf and n_cls == 2
        self.tabs.setTabVisible(1, binary)
        self.tabs.setTabVisible(2, clf)
        self.tabs.setTabVisible(4, not clf)
        self.t_res.set_frame(res.table(), stretch=False)
        cfg = res.config
        strat = (f"{cfg.n_splits} 折 × {cfg.n_repeats} 次重复交叉验证（共 {cfg.n_splits * cfg.n_repeats} 次拟合/模型）"
                 if cfg.strategy == "cv" else f"留出法（测试集 {cfg.test_size:.0%}）")
        extra = ""
        if clf:
            cnt = np.bincount(res.y, minlength=n_cls)
            extra = "&nbsp;&nbsp;|&nbsp;&nbsp;类别：" + "，".join(f"{res.classes[i]}={cnt[i]}" for i in range(n_cls))
        self.summary.setHtml(f"样本 n={len(res.y)}，特征 {len(res.feature_names)} 个&nbsp;&nbsp;|&nbsp;&nbsp;{strat}{extra}<br>"
                             f"表中为各折均值 ± 标准差；置信区间见“模型对比”图（折外预测 bootstrap 95% CI）。"
                             + ("" if b.in_fold_prep else "<br><b style='color:#C0392B'>注意：未在折内拟合预处理，结果可能偏乐观。</b>"))
        # 指标下拉
        tn = res.table_numeric()
        self.metric_box.blockSignals(True)
        self.metric_box.clear()
        for k in [c for c in tn.columns if c != "模型"]:
            self.metric_box.addItem(ev.METRIC_ZH.get(k, k), k)
        self.metric_box.setCurrentIndex(max(0, self.metric_box.findData(ev.default_metric(task))))
        self.metric_box.blockSignals(False)
        self._draw_compare()
        # 模型下拉
        self.cur.blockSignals(True)
        self.cur.clear()
        for r in res.results:
            if not r.error:
                self.cur.addItem(r.name, r.key)
        self.cur.setCurrentIndex(max(0, self.cur.findData(b.selected)))
        self.cur.blockSignals(False)
        if binary:
            items = self._items()
            ci = {r.name: r.ci for r in res.results if not r.error}
            self.p_roc.draw_with(pl.plot_roc, items, ci)
            self.p_pr.draw_with(pl.plot_pr, items)
            self.p_cal.draw_with(pl.plot_calibration, items)
            self.p_dca.draw_with(pl.plot_dca, items)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(b.threshold * 100)))
        self.slider.blockSignals(False)
        self._cur_changed()
        self.tabs.setCurrentIndex(0)

    def _items(self):
        res = self._bundle().res
        return [(r.name, res.y_oof(r), r.oof[:, 1]) for r in res.results if not r.error]

    def _draw_compare(self):
        b = self._bundle()
        if not b or self.metric_box.currentData() is None:
            return
        res = b.res
        ci = {r.name: r.ci for r in res.results if not r.error}
        self.p_cmp.draw_with(pl.plot_model_compare, res.table_numeric(), self.metric_box.currentData(), ci, self.metric_box.currentText())

    def _cur_changed(self):
        b, r = self._bundle(), self._cur_result()
        if not b or not r:
            return
        b.selected = r.key
        res, clf = b.res, b.task == "classification"
        y = res.y_oof(r)
        if clf:
            self._thr_changed()
        else:
            self.p_pv.draw_with(pl.plot_pred_vs_true, y, r.oof, r.name)
            self.p_resid.draw_with(pl.plot_residuals, y, r.oof)
        self.t_fold.set_frame(r.fold_scores.rename(columns=ev.METRIC_ZH).round(4) if r.fold_scores is not None else pd.DataFrame())
        spec = b.specs[r.key]
        full = {**{s.name: s.default for s in spec.params}, **r.params}
        self.params_view.setHtml(f"<b>{r.name}</b>　参数：{full}<br>耗时 {r.seconds:.2f}s；最终模型在全部 {len(res.y)} 个样本上重新拟合，用于预测页。")
        key = r.key
        # 已计算过的重要性直接显示
        if key in self._perm and self.imp_method.currentData() == "perm":
            self._show_perm(key)

    # ---------- 阈值 ----------
    def _thr_changed(self):
        b, r = self._bundle(), self._cur_result()
        if not b or not r or b.task != "classification":
            return
        res = b.res
        t = self.slider.value() / 100
        b.threshold = t
        self.thr_lbl.setText(f"{t:.2f}")
        y = res.y_oof(r)
        n = len(res.classes)
        if n == 2:
            p = r.oof[:, 1]
            m = binary_confusion_metrics(y, (p >= t).astype(int))
            self.thr_info.setText(
                f"灵敏度 {m['sensitivity']:.1%}（漏诊 FN={m['FN']}）　特异度 {m['specificity']:.1%}（误诊 FP={m['FP']}）　"
                f"PPV {m['ppv']:.1%}　NPV {m['npv']:.1%}　准确率 {m['accuracy']:.1%}　　"
                f"提示：筛查场景通常优先保证灵敏度（降低漏诊），代价是特异度下降。")
            cm = ev.conf_matrix(y, (p >= t).astype(int), labels=[0, 1])
            self.p_cm.draw_with(pl.plot_confusion, cm, res.classes, f"混淆矩阵（{r.name}，阈值 {t:.2f}）")
            self.p_thr.draw_with(pl.plot_threshold, ev.threshold_table(y, p), t)
        else:
            cm = ev.conf_matrix(y, r.oof.argmax(1), labels=list(range(n)))
            self.thr_info.setText("多分类任务：按概率最大的类别判定（阈值不适用）。")
            self.p_cm.draw_with(pl.plot_confusion, cm, res.classes, f"混淆矩阵（{r.name}）")
            self.p_thr.message("阈值曲线仅用于二分类")

    def _auto_thr(self, how: str):
        b, r = self._bundle(), self._cur_result()
        if not b or not r or len(b.res.classes) != 2:
            return
        y, p = b.res.y_oof(r), r.oof[:, 1]
        t = ev.youden_threshold(y, p) if how == "youden" else ev.threshold_for_sensitivity(y, p, 0.9)
        self.slider.setValue(int(np.clip(round(t * 100), 5, 95)))

    # ---------- 重要性 ----------
    def _importance(self):
        b, r = self._bundle(), self._cur_result()
        if not b or not r:
            self.error("尚无模型", "请先训练模型。")
            return
        method = self.imp_method.currentData()
        spec = b.specs[r.key]
        res = b.res
        key = r.key
        if method == "native":
            nat = ex.native_importance(r.pipeline)
            if nat is None:
                self.error("该模型没有自带的重要性", f"{r.name} 没有 feature_importances_ 或系数，请改用置换重要性或 SHAP。")
                return
            col = nat.columns[1]
            self.p_imp.draw_with(pl.plot_importance, nat, col, "特征", f"{r.name}：{col}")
            self.t_imp.set_frame(nat.round(4))
            b.importance = nat.rename(columns={"特征": "变量"})
            return
        if method == "perm":
            if key in self._perm:
                self._show_perm(key)
                return
            X = res.X
            cfg = res.config

            def job(progress, cancel):
                progress(0, 0, "计算折外置换重要性…")
                return ex.heldout_permutation_importance(X, res.y, spec, r.params, res.num_cols, res.cat_cols, cfg)

            def done(df):
                self._perm[key] = df
                self._show_perm(key)
            self.runner.start(job, done, busy_text="计算置换重要性…")
            return
        # SHAP
        if b.task == "classification" and len(res.classes) != 2:
            self.error("SHAP 暂不支持多分类", "请使用置换重要性。")
            return
        if key in self._shap:
            self._show_shap(key)
            return
        X = res.X

        def job(progress, cancel):
            progress(0, 0, "计算 SHAP 值…")
            return ex.compute_shap(r.pipeline, spec, X, b.task, len(res.classes))

        def done(sh):
            self._shap[key] = sh
            self._show_shap(key)
        self.runner.start(job, done, busy_text="计算 SHAP 值（核方法较慢，请稍候）…")

    def _show_perm(self, key):
        b = self._bundle()
        df = self._perm[key]
        b.importance = df
        self.p_imp.draw_with(pl.plot_importance, df, "重要性(得分下降)", "变量",
                          f"置换重要性：{b.res.get(key).name}（折外，得分下降越多越重要）", 15, "标准差(折间)")
        self.t_imp.set_frame(df.round(4))

    def _show_shap(self, key):
        b = self._bundle()
        sh = self._shap[key]
        b.shap = sh
        self.p_imp.draw_with(pl.plot_shap_summary, sh)
        self.t_imp.set_frame(sh.mean_abs().round(4))
        b.importance = sh.mean_abs().rename(columns={"特征": "变量", "mean|SHAP|": "重要性(mean|SHAP|)"})

    # ---------- 调参 ----------
    def _tune(self):
        if not self.need_data() or self.runner.busy():
            return
        it = self.algs.currentItem()
        if it is None:
            return
        spec = REGISTRY[it.data(Qt.UserRole)]
        prep = self._prepare()
        if prep is None:
            return
        X, y, classes, num, cat, types, tgt = prep
        cfg = self._config()

        def job(progress, cancel):
            progress(0, 0, f"调参：{spec.name}…")
            return ev.tune_model(spec, X, y, num, cat, cfg, n_iter=24)

        def done(out):
            best, score, tab = out
            dlg = QDialog(self)
            dlg.setWindowTitle(f"调参结果：{spec.name}")
            dlg.resize(640, 420)
            v = QVBoxLayout(dlg)
            v.addWidget(hint(f"最佳参数：{best}　　内层交叉验证得分 = {score:.4f}\n"
                             "注意：用同一份数据调参再评估会偏乐观；调参后请回到主界面重新做交叉验证对比，并以其结果为准。", warn=True))
            t = DataTable(highlight_nan=False)
            t.set_frame(tab.head(15).round(4))
            v.addWidget(t, 1)
            bb = QDialogButtonBox()
            ok = bb.addButton("应用最佳参数", QDialogButtonBox.AcceptRole)
            bb.addButton("关闭", QDialogButtonBox.RejectRole)
            bb.accepted.connect(dlg.accept)
            bb.rejected.connect(dlg.reject)
            v.addWidget(bb)
            if dlg.exec():
                self._params[spec.key] = {**self._params.get(spec.key, {}), **best}
                self.form.set_specs(spec.params, self._params[spec.key])
                it.setCheckState(Qt.Checked)
                self.toast(f"已应用 {spec.name} 的最佳参数")

        self.runner.start(job, done, busy_text=f"调参中：{spec.name}…")

    # ---------- 保存 / 报告 ----------
    def package_for(self, key: str | None = None):
        b = self._bundle()
        if not b:
            return None
        r = b.res.get(key or b.selected)
        spec = b.specs[r.key]
        return persist.make_package(r.name, r.pipeline, b.task, b.target, b.res.X, b.res.num_cols, b.res.cat_cols,
                                    b.res.classes, b.threshold, r.metrics, spec.key)

    def _save_model(self):
        b = self._bundle()
        if not b:
            self.error("尚无模型", "请先训练模型。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存模型", f"{self.cur.currentText()}.joblib", "MedMiner 模型 (*.joblib)")
        if path:
            persist.save_package(self.package_for(), path)
            self.toast(f"模型已保存：{path}（文件含模型参数，不含原始数据）")

    def _to_report(self):
        b = self._bundle()
        if not b:
            self.error("尚无结果", "请先训练模型。")
            return
        res = b.res
        self.state.report_add_table(res.table(), "模型评估结果（交叉验证）", "建模评估")
        ci = {r.name: r.ci for r in res.results if not r.error}
        metric = self.metric_box.currentData() or ev.default_metric(b.task)
        f = pl.new_figure(7, 3.6)
        pl.plot_model_compare(f, res.table_numeric(), metric, ci, ev.METRIC_ZH.get(metric, metric))
        self.state.report_add_figure(pl.fig_to_png(f), "模型对比", "建模评估")
        if b.task == "classification" and len(res.classes) == 2:
            f = pl.new_figure(6, 5)
            pl.plot_roc(f, self._items(), ci)
            self.state.report_add_figure(pl.fig_to_png(f), "ROC 曲线", "建模评估")
            r = self._cur_result()
            f = pl.new_figure(4.5, 4)
            y, p = res.y_oof(r), r.oof[:, 1]
            pl.plot_confusion(f, ev.conf_matrix(y, (p >= b.threshold).astype(int), [0, 1]), res.classes,
                              f"混淆矩阵（{r.name}，阈值 {b.threshold:.2f}）")
            self.state.report_add_figure(pl.fig_to_png(f), "混淆矩阵", "建模评估")
        if b.importance is not None:
            self.state.report_add_table(b.importance.head(15).round(4), "特征重要性", "建模评估")

