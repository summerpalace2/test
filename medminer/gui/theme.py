"""统一样式（QSS）与字体。颜色集中在 PALETTE，改色只改这里。"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from .icons import arrow_pixmap

PALETTE = dict(
    accent="#3B6EF6", accent_hover="#2A58D8", accent_soft="#E8EFFF",
    bg="#F4F6FB", card="#FFFFFF", border="#E3E8F2", border_strong="#CBD3E3",
    text="#1F2A44", muted="#6B778C", nav_top="#18264D", nav_bottom="#0F1A38",
    ok="#16A394", warn_bg="#FFF6E6", warn_text="#8A5300", warn_border="#F6DDAE",
)
ACCENT = PALETTE["accent"]


def _qss(P: dict, arrows: dict) -> str:
    return f"""
* {{ font-size: 13px; color: {P['text']}; }}
QMainWindow, QDialog {{ background: {P['bg']}; }}
QWidget#page {{ background: {P['bg']}; }}
QToolTip {{ background: {P['text']}; color: white; border: none; padding: 5px 8px; border-radius: 6px; }}

/* ---------- 侧边导航 ---------- */
QFrame#nav {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {P['nav_top']}, stop:1 {P['nav_bottom']}); }}
QLabel#brand {{ color: white; font-size: 18px; font-weight: bold; }}
QLabel#brandSub {{ color: #8FA3C9; font-size: 11px; }}
QLabel#navVersion {{ color: #5F7399; font-size: 11px; padding: 6px 18px; }}
QListWidget#navList {{ background: transparent; border: none; outline: none; padding: 4px 8px; }}
QListWidget#navList::item {{ color: #B9C7E3; padding: 10px 12px; margin: 2px 0; border-radius: 10px; }}
QListWidget#navList::item:hover {{ background: rgba(255,255,255,0.07); color: white; }}
QListWidget#navList::item:selected {{ background: rgba(91,140,255,0.28); color: white; font-weight: bold; }}

/* ---------- 文字 ---------- */
QLabel {{ background: transparent; }}
QLabel#h1 {{ font-size: 20px; font-weight: bold; padding-left: 10px; border-left: 4px solid {P['accent']}; }}
QLabel#hint {{ color: {P['muted']}; }}
QLabel#warn {{ background: {P['warn_bg']}; color: {P['warn_text']}; border: 1px solid {P['warn_border']}; border-radius: 8px; padding: 7px 12px; }}
QLabel#ok {{ background: #E7F6F3; color: #136457; border: 1px solid #BFE5DE; border-radius: 8px; padding: 7px 12px; }}

/* ---------- 卡片 ---------- */
QGroupBox {{ background: {P['card']}; border: 1px solid {P['border']}; border-radius: 12px; margin-top: 16px; padding: 14px 10px 10px 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 14px; padding: 0 6px; color: {P['text']}; font-weight: bold; }}

/* ---------- 按钮 ---------- */
QPushButton {{ background: {P['card']}; border: 1px solid {P['border_strong']}; border-radius: 8px; padding: 6px 16px; min-height: 20px; }}
QPushButton:hover {{ border-color: {P['accent']}; color: {P['accent']}; background: #FAFCFF; }}
QPushButton:pressed {{ background: {P['accent_soft']}; }}
QPushButton:disabled {{ color: #A3ADBF; background: #EEF1F6; border-color: {P['border']}; }}
QPushButton#primary {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5586FF, stop:1 {P['accent']});
    color: white; border: 1px solid {P['accent_hover']}; font-weight: bold; }}
QPushButton#primary:hover {{ background: {P['accent_hover']}; color: white; }}
QPushButton#primary:disabled {{ background: #A9BFEF; border-color: #A9BFEF; color: white; }}
QPushButton#danger {{ color: #C0392B; }}
QPushButton#danger:hover {{ border-color: #C0392B; background: #FFF5F4; color: #C0392B; }}
QPushButton::menu-indicator {{ image: none; width: 0; }}

/* ---------- 输入控件 ---------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{ background: {P['card']}; border: 1px solid {P['border_strong']};
    border-radius: 8px; padding: 4px 8px; min-height: 22px; selection-background-color: {P['accent']}; selection-color: white; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {P['accent']}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ background: #EEF1F6; color: #98A2B5; }}
QComboBox {{ padding-right: 26px; }}
QComboBox::drop-down {{ border: none; width: 24px; subcontrol-origin: padding; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url({arrows['down']}); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: white; border: 1px solid {P['border_strong']}; border-radius: 8px; padding: 4px;
    selection-background-color: {P['accent_soft']}; selection-color: {P['accent']}; outline: none; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 20px; border: none; border-top-right-radius: 8px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 20px; border: none; border-bottom-right-radius: 8px; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({arrows['up']}); width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({arrows['down']}); width: 10px; height: 10px; }}
QCheckBox, QRadioButton {{ spacing: 7px; }}

/* ---------- 列表 / 表格 ---------- */
QTableView, QTableWidget, QListWidget, QTextBrowser {{ background: {P['card']}; border: 1px solid {P['border']}; border-radius: 10px;
    gridline-color: #EEF1F6; alternate-background-color: #F8FAFE; selection-background-color: #D9E5FF; selection-color: {P['text']}; }}
QListWidget::item {{ padding: 4px 6px; }}
QHeaderView::section {{ background: #F1F4FA; color: #34476B; border: none; border-right: 1px solid {P['border']};
    border-bottom: 1px solid {P['border_strong']}; padding: 6px 8px; font-weight: bold; }}
QTableCornerButton::section {{ background: #F1F4FA; border: none; }}

/* ---------- 标签页 ---------- */
QTabWidget::pane {{ border: 1px solid {P['border']}; border-radius: 10px; background: {P['card']}; top: -1px; }}
QTabBar::tab {{ background: transparent; border: none; border-bottom: 2px solid transparent; padding: 9px 18px; margin-right: 4px; color: {P['muted']}; }}
QTabBar::tab:hover {{ color: {P['accent']}; }}
QTabBar::tab:selected {{ color: {P['accent']}; font-weight: bold; border-bottom: 2px solid {P['accent']}; }}

/* ---------- 其它 ---------- */
QProgressBar {{ border: none; border-radius: 6px; background: #E3E8F2; text-align: center; height: 10px; color: transparent; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5586FF, stop:1 #19B5A5); border-radius: 6px; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 8px; }}
QSplitter::handle:vertical {{ height: 8px; }}
QStatusBar {{ background: {P['card']}; border-top: 1px solid {P['border']}; color: {P['muted']}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
/* 注意：不能用 background: transparent，会让 matplotlib 工具栏图标消失（test_plot_toolbar_icons_visible 守护） */
QToolBar {{ background: #FAFBFE; border: none; spacing: 2px; }}
QSlider::groove:horizontal {{ height: 6px; background: #D5DDEB; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {P['accent']}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; border: 2px solid {P['accent']}; width: 14px; height: 14px; margin: -6px 0; border-radius: 9px; }}
QMenuBar {{ background: {P['card']}; border-bottom: 1px solid {P['border']}; }}
QMenuBar::item {{ padding: 5px 12px; background: transparent; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {P['accent_soft']}; }}
QMenu {{ background: white; border: 1px solid {P['border_strong']}; border-radius: 8px; padding: 5px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background: {P['accent_soft']}; color: {P['accent']}; }}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #C5CEDF; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:horizontal {{ background: #C5CEDF; border-radius: 4px; min-width: 30px; }}
QScrollBar::handle:hover {{ background: #A3B0CB; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""


def _asset_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "medminer_ui"
    d.mkdir(parents=True, exist_ok=True)
    return d


def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    families = set(QFontDatabase.families())
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
        if name in families:
            f = QFont(name)
            f.setPointSize(10)
            app.setFont(f)
            break
    # QSS 的 image:url() 需要文件路径，运行时把箭头图标画到临时目录
    arrows = {}
    for d in ("up", "down"):
        path = _asset_dir() / f"arrow_{d}.png"
        arrow_pixmap(d).save(str(path), "PNG")
        arrows[d] = path.as_posix()
    app.setStyleSheet(_qss(PALETTE, arrows))
