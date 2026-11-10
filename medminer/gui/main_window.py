"""主窗口：左侧导航 + 页面栈 + 菜单 + 状态栏 + 拖拽打开。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QStackedWidget, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..core import io as mio
from .pages.data_page import DataPage
from .state import AppState


def page_classes():
    """按导航顺序返回页面类。延迟导入，便于某个页面出错时不影响其它页面启动。"""
    from .pages.ai_page import AIPage
    from .pages.cluster_page import ClusterPage
    from .pages.image_page import ImagePage
    from .pages.model_page import ModelPage
    from .pages.predict_page import PredictPage
    from .pages.preprocess_page import PreprocessPage
    from .pages.report_page import ReportPage
    from .pages.stats_page import StatsPage
    from .pages.visualize_page import VisualizePage
    return [DataPage, PreprocessPage, VisualizePage, StatsPage, ModelPage, ClusterPage, PredictPage, ImagePage, AIPage, ReportPage]


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None, classes=None):
        super().__init__()
        self.state = state or AppState()
        self.setWindowTitle(f"{APP_NAME}  v{__version__}")
        self.resize(1380, 860)
        self.setMinimumSize(1100, 700)
        self.setAcceptDrops(True)

        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        nav = QFrame()
        nav.setObjectName("nav")
        nav.setFixedWidth(176)
        nl = QVBoxLayout(nav)
        nl.setContentsMargins(0, 0, 0, 8)
        nl.setSpacing(0)
        b = QLabel("MedMiner")
        b.setObjectName("brand")
        bs = QLabel("医学数据挖掘平台")
        bs.setObjectName("brandSub")
        nl.addWidget(b)
        nl.addWidget(bs)
        self.nav = QListWidget()
        self.nav.setObjectName("navList")
        nl.addWidget(self.nav, 1)
        h.addWidget(nav)
        self.stack = QStackedWidget()
        h.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.pages: dict[str, QWidget] = {}
        for cls in (classes if classes is not None else page_classes()):
            self._add_page(cls)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)
        self.state.goto_page.connect(self.goto)
        self.state.status.connect(lambda m: self.statusBar().showMessage(m, 8000))
        self.state.dataset_changed.connect(self._title)
        self._menu()
        self.statusBar().showMessage("就绪：请先打开数据文件，或加载示例数据开始体验。")
        QSettings("MedMiner", "MedMiner")  # 预留：窗口状态持久化

    def _add_page(self, cls):
        p = cls(self.state)
        self.pages[p.key] = p
        self.stack.addWidget(p)
        it = QListWidgetItem(f"  {self.nav.count() + 1}  {p.title}")
        self.nav.addItem(it)

    def goto(self, key: str):
        if key in self.pages:
            self.nav.setCurrentRow(list(self.pages).index(key))

    def _title(self):
        ds = self.state.ds
        self.setWindowTitle(f"{ds.name} - {APP_NAME}" if ds else APP_NAME)

    # ---------- 菜单 ----------
    def _menu(self):
        mb = self.menuBar()
        f = mb.addMenu("文件(&F)")
        data = self.pages.get("data")

        def act(menu, text, fn, key=None):
            a = QAction(text, self)
            if key:
                a.setShortcut(QKeySequence(key))
            a.triggered.connect(lambda: fn())
            menu.addAction(a)

        if data:
            act(f, "打开数据文件…", data.open_file, "Ctrl+O")
            act(f, "加载示例数据", data.load_sample)
            f.addSeparator()
            act(f, "打开项目…", data.open_project)
            act(f, "保存项目…", data.save_project, "Ctrl+S")
            f.addSeparator()
        act(f, "退出", self.close, "Ctrl+Q")
        hm = mb.addMenu("帮助(&H)")
        act(hm, "使用提示", self._tips)
        act(hm, "关于", self._about)

    def _tips(self):
        QMessageBox.information(self, "使用提示", (
            "<b>推荐流程</b><br>1. 数据管理：打开数据，检查列类型，指定目标列<br>"
            "2. 数据预处理：缺失值/异常值/二值化目标…（可保存为管道）<br>3. 可视化 / 统计分析：了解数据，生成基线表<br>"
            "4. 建模评估：多模型对比（默认在交叉验证折内做预处理，避免数据泄漏）<br>5. 聚类 / 预测：亚型发现、单个样本风险预测<br>"
            "6. AI 分析：配置大模型后一键解读各步结果<br>7. 报告导出：HTML / Word / PDF / Markdown<br><br>"
            "<b>小样本提示</b>：样本量几百时，请以交叉验证均值±标准差和置信区间为准，不要只看一次划分的结果。"))

    def _about(self):
        QMessageBox.about(self, "关于", f"<b>{APP_NAME}</b><br>版本 {__version__}<br><br>"
                          "面向医学数据挖掘课程实验的桌面分析软件：数据预处理、可视化、统计、回归/分类/聚类、"
                          "可解释性（SHAP）与大模型辅助分析。<br><br>仅用于教学与研究，不构成临床诊断或治疗建议。")

    # ---------- 拖拽 ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir() or p.suffix.lower() in mio.IMAGE_EXTS:
                img = self.pages.get("image")
                if img:
                    self.goto("image")
                    img.open_folder(str(p if p.is_dir() else p.parent))
            elif p.suffix.lower() in mio.TABLE_EXTS or p.suffix.lower() == ".medproj":
                data = self.pages.get("data")
                self.goto("data")
                if p.suffix.lower() == ".medproj":
                    from ..core.dataset import load_project
                    self.state.set_dataset(load_project(p))
                else:
                    data.open_file(str(p))
            break
