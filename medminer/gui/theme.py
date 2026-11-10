"""统一样式（QSS）与字体。"""
from __future__ import annotations

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

ACCENT = "#2F6FDE"
NAV_BG = "#142038"

QSS = f"""
* {{ font-size: 13px; }}
QMainWindow, QDialog {{ background: #F4F6FA; }}
QWidget#page {{ background: #F4F6FA; }}
QFrame#nav {{ background: {NAV_BG}; }}
QLabel#brand {{ color: white; font-size: 17px; font-weight: bold; padding: 18px 16px 2px 16px; }}
QLabel#brandSub {{ color: #8FA3C7; font-size: 11px; padding: 0 16px 14px 16px; }}
QListWidget#navList {{ background: transparent; border: none; outline: none; }}
QListWidget#navList::item {{ color: #C5D1E8; padding: 11px 18px; border-left: 3px solid transparent; }}
QListWidget#navList::item:hover {{ background: #1D2E4F; }}
QListWidget#navList::item:selected {{ background: #223A66; color: white; border-left: 3px solid {ACCENT}; }}
QLabel#h1 {{ font-size: 18px; font-weight: bold; color: #1B2A47; }}
QLabel#hint {{ color: #6B7A90; }}
QLabel#warn {{ background: #FFF4E5; color: #8A5300; border: 1px solid #F5D9A8; border-radius: 6px; padding: 6px 10px; }}
QLabel#ok {{ background: #E8F6F3; color: #146356; border: 1px solid #BDE5DD; border-radius: 6px; padding: 6px 10px; }}
QGroupBox {{ background: white; border: 1px solid #E1E6EF; border-radius: 8px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #3B4B6B; font-weight: bold; }}
QPushButton {{ background: white; border: 1px solid #C9D2E3; border-radius: 6px; padding: 6px 14px; color: #1B2A47; }}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:disabled {{ color: #A3ADBF; background: #EEF1F6; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border: 1px solid {ACCENT}; font-weight: bold; }}
QPushButton#primary:hover {{ background: #2459B5; color: white; }}
QPushButton#primary:disabled {{ background: #9DB6E8; border-color: #9DB6E8; color: white; }}
QPushButton#danger {{ color: #C0392B; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{ background: white; border: 1px solid #C9D2E3;
    border-radius: 5px; padding: 3px 6px; selection-background-color: {ACCENT}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{ border-color: {ACCENT}; }}
QTableView, QTableWidget, QListWidget, QTextBrowser {{ background: white; border: 1px solid #E1E6EF; border-radius: 6px;
    gridline-color: #EEF1F6; alternate-background-color: #F8FAFD; selection-background-color: #D6E4FF; selection-color: #14213D; }}
QHeaderView::section {{ background: #EEF3FB; color: #34476B; border: none; border-right: 1px solid #E1E6EF;
    border-bottom: 1px solid #D5DDEB; padding: 5px 8px; font-weight: bold; }}
QTabWidget::pane {{ border: 1px solid #E1E6EF; border-radius: 6px; background: white; top: -1px; }}
QTabBar::tab {{ background: #E9EEF7; border: 1px solid #E1E6EF; border-bottom: none; padding: 7px 16px; margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px; color: #4A5B7C; }}
QTabBar::tab:selected {{ background: white; color: {ACCENT}; font-weight: bold; }}
QProgressBar {{ border: 1px solid #C9D2E3; border-radius: 5px; background: white; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QSplitter::handle {{ background: #E1E6EF; }}
QStatusBar {{ background: white; border-top: 1px solid #E1E6EF; color: #4A5B7C; }}
QScrollArea {{ border: none; background: transparent; }}
QToolBar {{ background: transparent; border: none; spacing: 2px; }}
QSlider::groove:horizontal {{ height: 6px; background: #D5DDEB; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 16px; margin: -5px 0; border-radius: 8px; }}
QMenuBar {{ background: white; }} QMenu {{ background: white; border: 1px solid #C9D2E3; }}
QMenu::item:selected {{ background: #D6E4FF; color: #14213D; }}
"""


def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    families = set(QFontDatabase.families())
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
        if name in families:
            f = QFont(name)
            f.setPointSize(10)
            app.setFont(f)
            break
    app.setStyleSheet(QSS)
