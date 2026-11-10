"""⑩ 报告：管理要放进报告的内容，预览并导出 HTML / Word / PDF / Markdown。"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget)

from ...core import cluster as cl
from ...core import evaluate as ev
from ...core import plots as pl
from ...core import stats
from ...core.report import Report
from .base import Page, group, h1, hint


def build_auto_report(state) -> int:
    """根据当前所有已有结果自动生成一份完整报告（追加到现有条目之后）。返回新增条目数。"""
    r: Report = state.report
    n0 = len(r.items)
    ds = state.ds
    if ds is None:
        return 0
    df, types = ds.current(), ds.current_types()
    r.add_heading("一、数据概况")
    r.add_text(f"数据集 **{ds.name}**：{ds.raw.shape[0]} 例 × {ds.raw.shape[1]} 个变量；目标列：{ds.target or '未指定'}；"
               f"缺失单元格 {int(ds.raw.isna().sum().sum())} 个（{ds.raw.isna().mean().mean() * 100:.2f}%）。")
    r.add_table(stats.describe(df, types), "描述统计")
    if df.isna().values.any():
        f = pl.new_figure(7, 3.6)
        pl.draw_chart(f, "missing", df, types)
        r.add_figure(pl.fig_to_png(f), "缺失值分布")
    r.add_heading("二、数据预处理")
    log = ds.log()
    if log:
        r.add_table(pd.DataFrame([{"步骤": e["index"] + 1, "操作": e["desc"], "状态": e["status"], "数据形状变化": e["msg"]} for e in log]),
                    "预处理步骤")
    else:
        r.add_text("未进行额外预处理（模型内部仍会对缺失值进行填充）。")
    c = state.cache
    if "table1" in c or "or" in c:
        r.add_heading("三、统计分析")
        if "table1" in c:
            r.add_table(c["table1"], f"基线特征表（按 {c.get('table1_group', '')} 分组）")
        if "or" in c:
            r.add_table(c["or"].round(4), "逻辑回归比值比 (OR)")
            f = pl.new_figure(6, 4)
            pl.plot_forest(f, c["or"])
            r.add_figure(pl.fig_to_png(f), "OR 森林图")
    b = state.eval
    if b is not None:
        res = b.res
        r.add_heading("四、模型评估")
        cfg = res.config
        strat = f"{cfg.n_splits} 折 × {cfg.n_repeats} 次重复交叉验证" if cfg.strategy == "cv" else f"留出法（测试集 {cfg.test_size:.0%}）"
        r.add_text(f"任务：{'分类' if b.task == 'classification' else '回归'}；目标：{b.target}；样本 n={len(res.y)}；特征 {len(res.feature_names)} 个；"
                   f"验证：{strat}；预处理{'在每折内拟合（无泄漏）' if b.in_fold_prep else '在全量数据上拟合（可能偏乐观）'}。"
                   "表中为各折均值 ± 标准差。")
        r.add_table(res.table(), "模型评估结果")
        metric = ev.default_metric(b.task)
        ci = {x.name: x.ci for x in res.results if not x.error}
        f = pl.new_figure(7, 3.6)
        pl.plot_model_compare(f, res.table_numeric(), metric, ci, ev.METRIC_ZH[metric])
        r.add_figure(pl.fig_to_png(f), f"模型对比（{ev.METRIC_ZH[metric]}，含 95% CI）")
        if b.task == "classification" and len(res.classes) == 2:
            items = [(x.name, res.y_oof(x), x.oof[:, 1]) for x in res.results if not x.error]
            f = pl.new_figure(6, 5)
            pl.plot_roc(f, items, ci)
            r.add_figure(pl.fig_to_png(f), "ROC 曲线（折外预测）")
            cur = res.get(b.selected)
            f = pl.new_figure(4.5, 4)
            y, p = res.y_oof(cur), cur.oof[:, 1]
            pl.plot_confusion(f, ev.conf_matrix(y, (p >= b.threshold).astype(int), [0, 1]), res.classes,
                              f"混淆矩阵（{cur.name}，阈值 {b.threshold:.2f}）")
            r.add_figure(pl.fig_to_png(f), "混淆矩阵")
        if b.importance is not None:
            r.add_table(b.importance.head(15).round(4), "特征重要性")
        if b.shap is not None:
            f = pl.new_figure(7, 5)
            pl.plot_shap_summary(f, b.shap)
            r.add_figure(pl.fig_to_png(f), "SHAP 汇总图")
    if state.cluster is not None:
        cb = state.cluster
        r.add_heading("五、聚类分析")
        res = cb.res
        r.add_text(f"算法：{res.method}；簇数 {res.n_clusters}；各簇样本数 {res.sizes}；"
                   + "；".join(f"{k}={v:.3f}" for k, v in res.metrics.items()))
        emb, var = cl.embed_2d(res.Xt)
        f = pl.new_figure(7, 4.5)
        pl.plot_cluster_scatter(f, emb, res.labels, var, "PC")
        r.add_figure(pl.fig_to_png(f), "聚类结果（PCA 投影）")
        if cb.profile is not None:
            r.add_table(cb.profile, "簇画像")
    if state.ai_notes:
        r.add_heading("六、AI 辅助解读")
        for title, text in state.ai_notes:
            r.add_ai(text, title[:30])
    return len(r.items) - n0


class ReportPage(Page):
    key, title = "report", "报告导出"

    def __init__(self, state):
        super().__init__(state)
        lay = self.root_layout()
        lay.addWidget(h1("报告导出"))
        lay.addWidget(hint("各页面的“加入报告”按钮会把表格/图表/AI 解读收集到这里；也可以“一键生成完整报告”。可调整顺序、增删，再导出。"))
        meta = QHBoxLayout()
        self.title_edit = QLineEdit(state.report.title)
        self.author_edit = QLineEdit(state.report.author)
        self.author_edit.setPlaceholderText("作者/小组（选填）")
        meta.addWidget(QLabel("标题"))
        meta.addWidget(self.title_edit, 2)
        meta.addWidget(QLabel("作者"))
        meta.addWidget(self.author_edit, 1)
        lay.addLayout(meta)
        self.title_edit.textChanged.connect(self._meta)
        self.author_edit.textChanged.connect(self._meta)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)
        self.items = QListWidget()
        self.items.setAlternatingRowColors(True)
        ll.addWidget(QLabel("报告内容（按顺序）"))
        ll.addWidget(self.items, 1)
        r1 = QHBoxLayout()
        for text, fn in (("↑", lambda: self._move(-1)), ("↓", lambda: self._move(1)), ("删除", self._remove), ("清空", self._clear)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            if text in ("删除", "清空"):
                b.setObjectName("danger")
            r1.addWidget(b)
        ll.addLayout(r1)
        r2 = QHBoxLayout()
        b1, b2 = QPushButton("＋ 章节标题"), QPushButton("＋ 文字段落")
        b1.clicked.connect(self._add_heading)
        b2.clicked.connect(self._add_text)
        r2.addWidget(b1)
        r2.addWidget(b2)
        ll.addLayout(r2)
        auto = QPushButton("一键生成完整报告")
        auto.setObjectName("primary")
        auto.clicked.connect(self._auto)
        ll.addWidget(auto)
        ex = QHBoxLayout()
        for text, fn in (("导出 HTML", self._export_html), ("导出 Word", self._export_docx), ("导出 PDF", self._export_pdf),
                         ("导出 Markdown", self._export_md)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            ex.addWidget(b)
        ll.addWidget(group("导出", ex))
        split.addWidget(left)
        self.preview = QTextBrowser()
        split.addWidget(self.preview)
        split.setSizes([380, 820])
        lay.addWidget(split, 1)
        state.report_changed.connect(self.refresh)
        self.refresh()

    def _meta(self):
        self.state.report.title = self.title_edit.text() or "分析报告"
        self.state.report.author = self.author_edit.text()
        self._preview()

    def refresh(self):
        self.items.clear()
        for it in self.state.report.items:
            self.items.addItem(QListWidgetItem(it.label() + (f"　〔{it.source}〕" if it.source else "")))
        self._preview()

    def _preview(self):
        if self.state.report.items:
            self.preview.setHtml(self.state.report.to_html(for_qt=True))
        else:
            self.preview.setHtml("<h3 style='color:#6B7A90'>报告还是空的</h3><p>在各页面点击“加入报告”，或点左侧“一键生成完整报告”。</p>")

    def _move(self, d):
        i = self.items.currentRow()
        if i >= 0 and 0 <= i + d < len(self.state.report.items):
            self.state.report.move(i, i + d)
            self.refresh()
            self.items.setCurrentRow(i + d)

    def _remove(self):
        i = self.items.currentRow()
        if i >= 0:
            self.state.report.remove(i)
            self.refresh()

    def _clear(self):
        if self.state.report.items and QMessageBox.question(self, "清空报告", "确定清空报告中的全部内容？") == QMessageBox.Yes:
            self.state.report.items.clear()
            self.state.ai_notes.clear()
            self.refresh()

    def _add_heading(self):
        t, ok = QInputDialog.getText(self, "章节标题", "标题：")
        if ok and t.strip():
            self.state.report.add_heading(t.strip())
            self.refresh()

    def _add_text(self):
        t, ok = QInputDialog.getMultiLineText(self, "文字段落", "内容（支持 Markdown）：")
        if ok and t.strip():
            self.state.report.add_text(t.strip())
            self.refresh()

    def _auto(self):
        if not self.need_data():
            return
        if self.state.report.items and QMessageBox.question(
                self, "一键生成", "报告里已有内容，新生成的内容会追加在后面。继续？") != QMessageBox.Yes:
            return
        try:
            n = build_auto_report(self.state)
        except Exception as e:
            self.error("生成失败", f"{type(e).__name__}: {e}")
            return
        self.refresh()
        self.toast(f"已生成 {n} 个条目" + ("" if self.state.eval else "（提示：尚未训练模型，报告中没有模型部分）"))

    # ---------- 导出 ----------
    def _export(self, caption, default, flt, fn):
        if not self.state.report.items:
            self.error("报告为空", "请先添加内容。")
            return
        path, _ = QFileDialog.getSaveFileName(self, caption, default, flt)
        if not path:
            return
        try:
            fn(path)
        except Exception as e:
            self.error("导出失败", f"{type(e).__name__}: {e}")
            return
        self.toast(f"已导出：{path}")

    def _export_html(self):
        self._export("导出 HTML", "report.html", "HTML (*.html)", self.state.report.save_html)

    def _export_docx(self):
        self._export("导出 Word", "report.docx", "Word (*.docx)", self.state.report.save_docx)

    def _export_pdf(self):
        self._export("导出 PDF", "report.pdf", "PDF (*.pdf)", self.state.report.save_pdf)

    def _export_md(self):
        self._export("导出 Markdown", "report.md", "Markdown (*.md)", self.state.report.save_markdown)
