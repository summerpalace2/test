# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置。
#   文件夹版（默认，启动快）：build_windows.bat        -> dist/MedMiner/MedMiner.exe（需整个文件夹一起拷贝）
#   单文件版（只有一个 exe）：build_windows_onefile.bat -> dist/MedMiner.exe（每次启动先解压到临时目录，较慢）
import os

ONEFILE = os.environ.get("MEDMINER_ONEFILE") == "1"
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, collect_submodules

datas, binaries, hiddenimports = [], [], []


def _wanted(name: str) -> bool:
    """排除测试/基准子模块。注意不能用 collect_all：它会导入 xgboost.testing，
    而其中的 pytest.importorskip 抛出的 Skipped 是 BaseException，会直接中断打包。"""
    return not any(p in ("tests", "testing", "test", "benchmark", "conftest") for p in name.split("."))


# 这些包带有数据文件/动态库/按需导入的子模块，必须完整收集
for pkg in ("shap", "lightgbm", "xgboost", "sklearn", "statsmodels", "seaborn", "docx", "openpyxl", "markdown"):
    hiddenimports += collect_submodules(pkg, filter=_wanted)
    datas += collect_data_files(pkg, excludes=["**/tests/**", "**/testing/**"])
    binaries += collect_dynamic_libs(pkg)

datas += collect_data_files("matplotlib")
datas += [("resources/sample", "resources/sample")]  # 内置示例数据
hiddenimports += collect_submodules("scipy.special") + ["scipy._lib.messagestream"]
hiddenimports += ["PySide6.QtSvg", "matplotlib.backends.backend_qtagg", "medminer.gui.pages.data_page",
                  "medminer.gui.pages.preprocess_page", "medminer.gui.pages.visualize_page",
                  "medminer.gui.pages.stats_page", "medminer.gui.pages.model_page", "medminer.gui.pages.cluster_page",
                  "medminer.gui.pages.predict_page", "medminer.gui.pages.image_page", "medminer.gui.pages.ai_page",
                  "medminer.gui.pages.report_page"]  # 主窗口里是延迟导入，需显式声明

a = Analysis(
    ["run.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "IPython", "notebook", "jupyter", "pytest", "PyQt5", "PyQt6", "wx", "torch", "tensorflow",
              "numba.tests", "sklearn.tests", "scipy.tests", "pandas.tests"],
    noarchive=False,
)
pyz = PYZ(a.pure)
if ONEFILE:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="MedMiner", console=False, debug=False, strip=False, upx=False,
              runtime_tmpdir=None)
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MedMiner", console=False, debug=False, strip=False, upx=False)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="MedMiner")
