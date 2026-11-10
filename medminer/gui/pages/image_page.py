"""⑧ 图像数据：浏览图像文件夹、提取图像特征并转成可分析的数据集。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem, QProgressBar,
                               QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ...core import images as im
from ...core.dataset import Dataset
from ...core.evaluate import Cancelled
from ..widgets.tables import DataTable
from ..widgets.tasks import TaskRunner
from .base import Page, h1, hint

MAX_THUMBS = 400


class ImagePage(Page):
    key, title = "image", "图像数据"

    def __init__(self, state):
        super().__init__(state)
        self.info: pd.DataFrame | None = None
        self.folder: str | None = None
        lay = self.root_layout()
        lay.addWidget(h1("图像数据"))
        lay.addWidget(hint("选择图像文件夹（子文件夹名会作为类别标签，如 normal / lesion）。可浏览缩略图，并提取灰度统计、熵、边缘、"
                           "中心-边缘对比等特征，生成数据集后即可使用预处理、统计、建模、聚类等全部功能。"
                           "支持 PNG/JPG/BMP/TIFF/GIF/WebP；DICOM 请先导出为 PNG/TIFF。"))
        bar = QHBoxLayout()
        self.btn_open = QPushButton("选择图像文件夹…")
        self.btn_open.setObjectName("primary")
        self.btn_open.clicked.connect(lambda: self.open_folder())
        self.btn_extract = QPushButton("提取特征 → 生成数据集")
        self.btn_extract.setEnabled(False)
        self.btn_extract.clicked.connect(self._extract)
        bar.addWidget(self.btn_open)
        bar.addWidget(self.btn_extract)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.summary = hint("尚未选择文件夹。")
        lay.addWidget(self.summary)
        self.bar = QProgressBar()
        lay.addWidget(self.bar)

        split = QSplitter(Qt.Horizontal)
        self.tabs = QTabWidget()
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListView.IconMode)
        self.gallery.setIconSize(QSize(96, 96))
        self.gallery.setResizeMode(QListView.Adjust)
        self.gallery.setSpacing(8)
        self.gallery.setWordWrap(True)
        self.gallery.setMovement(QListView.Static)
        self.gallery.currentItemChanged.connect(self._selected)
        self.tabs.addTab(self.gallery, "缩略图")
        self.table = DataTable(highlight_nan=False)
        self.tabs.addTab(self.table, "文件清单")
        split.addWidget(self.tabs)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.preview = QLabel("选择左侧图像以预览")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(300, 300)
        self.preview.setStyleSheet("background:white;border:1px solid #E1E6EF;border-radius:6px")
        rl.addWidget(self.preview, 1)
        self.feat_table = DataTable(highlight_nan=False)
        self.feat_table.setMaximumHeight(260)
        rl.addWidget(self.feat_table)
        split.addWidget(right)
        split.setSizes([700, 420])
        lay.addWidget(split, 1)
        self.runner = TaskRunner(self, self.bar, self.summary, self.btn_extract)

    def open_folder(self, folder: str | None = None):
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
            if not folder:
                return
        info = im.scan_image_folder(folder)
        if info.empty:
            self.error("没有找到图像", f"{folder}\n中没有可读取的图像文件（PNG/JPG/BMP/TIFF/GIF/WebP）。")
            return
        self.info, self.folder = info, folder
        labels = info["label"].replace("", "（无标签）").value_counts()
        self.summary.setText(f"{Path(folder).name}：{len(info)} 张图像　尺寸 {info['width'].min()}–{info['width'].max()} × "
                             f"{info['height'].min()}–{info['height'].max()}　类别：" + "，".join(f"{k}={v}" for k, v in labels.items())
                             + (f"　（缩略图仅显示前 {MAX_THUMBS} 张）" if len(info) > MAX_THUMBS else ""))
        self.table.set_frame(info.drop(columns="path"))
        self.gallery.clear()
        for r in info.head(MAX_THUMBS).itertuples(index=False):
            pm = QPixmap(r.path)
            if pm.isNull():
                continue
            it = QListWidgetItem(QIcon(pm.scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)),
                                 f"{r.label + '/' if r.label else ''}{Path(r.file).name}")
            it.setData(Qt.UserRole, r.path)
            it.setSizeHint(QSize(120, 130))
            self.gallery.addItem(it)
        self.btn_extract.setEnabled(True)
        if self.gallery.count():
            self.gallery.setCurrentRow(0)

    def _selected(self, cur, prev):
        if cur is None:
            return
        path = cur.data(Qt.UserRole)
        pm = QPixmap(path)
        if not pm.isNull():
            self.preview.setPixmap(pm.scaled(self.preview.size() - QSize(12, 12), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        try:
            f = im.image_features(path)
            self.feat_table.set_frame(pd.DataFrame({"特征": list(f), "取值": [round(v, 4) for v in f.values()]}))
        except Exception as e:
            self.feat_table.set_frame(pd.DataFrame({"提示": [f"无法提取特征：{e}"]}))

    def _extract(self):
        if self.info is None or self.runner.busy():
            return
        info, folder = self.info, self.folder

        def job(progress, cancel):
            def step(done, total):
                if cancel():
                    raise Cancelled()
                progress(done, total, f"提取特征 {done}/{total}")
            return im.extract_features(info, 128, step)

        def done(feats: pd.DataFrame):
            name = f"图像特征_{Path(folder).name}"
            ds = Dataset.from_frame(name, feats, {"image_folder": folder})
            if "label" in feats.columns:
                ds.target = "label"
            self.state.set_dataset(ds)
            self.state.goto_page.emit("data")
            self.toast(f"已生成数据集“{name}”：{len(feats)} 张图像 × {feats.shape[1]} 列（含 {len(im.FEATURE_NAMES)} 个图像特征）")

        self.runner.start(job, done, busy_text="提取特征…")
