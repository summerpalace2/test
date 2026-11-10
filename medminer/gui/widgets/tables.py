"""表格控件：DataFrame 模型 + 带右键菜单（复制/导出）的视图。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QKeySequence
from PySide6.QtWidgets import QAbstractItemView, QFileDialog, QHeaderView, QMenu, QTableView

_NAN_BG = QBrush(QColor("#FDE8E4"))


def fmt_value(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return ""
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else f"{v:.6g}"
    return str(v)


class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None, headers: dict[str, str] | None = None, highlight_nan: bool = True):
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()
        self._headers = headers or {}
        self._hl = highlight_nan

    def set_frame(self, df: pd.DataFrame, headers: dict[str, str] | None = None):
        self.beginResetModel()
        self._df = df
        self._headers = headers or {}
        self.endResetModel()

    def frame(self) -> pd.DataFrame:
        return self._df

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        v = self._df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            return fmt_value(v)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) \
                else int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.BackgroundRole and self._hl and fmt_value(v) == "" and (v is None or v is pd.NA or (isinstance(v, float))):
            return _NAN_BG
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                c = str(self._df.columns[section])
                return self._headers.get(c, c)
            return str(section + 1)
        return None

    def sort(self, column, order=Qt.AscendingOrder):
        if self._df.empty:
            return
        self.layoutAboutToBeChanged.emit()
        col = self._df.columns[column]
        try:
            self._df = self._df.sort_values(col, ascending=order == Qt.AscendingOrder, kind="stable", na_position="last")
        except TypeError:
            self._df = self._df.assign(_k=self._df[col].astype(str)).sort_values("_k", ascending=order == Qt.AscendingOrder).drop(columns="_k")
        self.layoutChanged.emit()


class DataTable(QTableView):
    """只读表格，支持排序、复制选区、导出 CSV。"""

    def __init__(self, parent=None, highlight_nan: bool = True):
        super().__init__(parent)
        self.m = PandasModel(highlight_nan=highlight_nan)
        self.setModel(self.m)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.verticalHeader().setDefaultSectionSize(24)
        self.horizontalHeader().setSortIndicatorShown(True)
        self.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)  # 初始不显示排序箭头

    def set_frame(self, df: pd.DataFrame, headers: dict[str, str] | None = None, stretch: bool = False):
        self.m.set_frame(df.reset_index(drop=True), headers)
        hh = self.horizontalHeader()
        if stretch:
            hh.setSectionResizeMode(QHeaderView.Stretch)
        else:
            hh.setSectionResizeMode(QHeaderView.Interactive)
            if df.shape[1] <= 40:
                self.resizeColumnsToContents()
                for i in range(df.shape[1]):  # 限宽，避免超长内容撑爆
                    self.setColumnWidth(i, min(max(self.columnWidth(i), 60), 360))

    def frame(self) -> pd.DataFrame:
        return self.m.frame()

    def copy_selection(self):
        idx = self.selectedIndexes()
        if not idx:
            return
        rows = sorted({i.row() for i in idx})
        cols = sorted({i.column() for i in idx})
        lines = ["\t".join(str(self.m.headerData(c, Qt.Horizontal)) for c in cols)]
        for r in rows:
            lines.append("\t".join(fmt_value(self.m.frame().iat[r, c]) for c in cols))
        QGuiApplication.clipboard().setText("\n".join(lines))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出为 CSV", "table.csv", "CSV (*.csv);;Excel (*.xlsx)")
        if not path:
            return
        df = self.m.frame()
        if path.lower().endswith(".xlsx"):
            df.to_excel(path, index=False)
        else:
            df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig：Excel 打开中文不乱码

    def keyPressEvent(self, e):
        if e.matches(QKeySequence.Copy):
            self.copy_selection()
        else:
            super().keyPressEvent(e)

    def _menu(self, pos):
        m = QMenu(self)
        m.addAction("复制选中单元格", self.copy_selection)
        m.addAction("导出整表为 CSV/Excel…", self.export_csv)
        m.exec(self.viewport().mapToGlobal(pos))
