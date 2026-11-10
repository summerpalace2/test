"""程序入口：python -m medminer.main  或  python main.py"""
from __future__ import annotations

import os
import sys


def selftest() -> int:
    """无界面自检：跑一遍核心流程，用于验证安装/打包是否完整（MedMiner.exe --selftest）。
    结果同时写入当前目录的 selftest_result.txt（打包后的 exe 没有控制台）。"""
    import tempfile
    import traceback
    from pathlib import Path

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.setdefault("QT_API", "pyside6")
    lines: list[str] = []
    ok = True
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # 全程只创建一个 QApplication（PDF 导出与界面都需要）

    def step(name, fn):
        nonlocal ok
        try:
            fn()
            lines.append(f"[ OK ] {name}")
        except BaseException as e:  # 包括 Skipped 等 BaseException
            ok = False
            lines.append(f"[FAIL] {name}: {type(e).__name__}: {e}")
            lines.append(traceback.format_exc())

    ctx: dict = {}

    def s_data():
        from .core.dataset import Dataset
        from .core.preprocess import Step
        from .core.sample_data import make_heart_sample
        ds = Dataset.from_frame("selftest", make_heart_sample())
        ds.add_step(Step("binarize", {"column": "num", "threshold": 0, "new_name": "dx"}))
        ctx["ds"] = ds

    def s_models():
        from .core import evaluate as ev
        from .core.features import prepare_xy
        from .core.models import REGISTRY
        ds = ctx["ds"]
        df, t = ds.base(), ds.base_types()
        X, y, cl, num, cat = prepare_xy(df, "dx", [c for c in df.columns if c not in ("num", "dx")], t, "classification")
        specs = [(REGISTRY[k], {"n_estimators": 30} if k in ("rf", "xgb", "lgbm") else {}) for k in ("logreg", "rf", "xgb", "lgbm", "svm")]
        res = ev.evaluate_models(X, y, cl, num, cat, specs, ev.EvalConfig(n_splits=3, n_repeats=1, n_boot=20))
        bad = [r.name + ":" + r.error for r in res.results if r.error]
        assert not bad, bad
        ctx["res"], ctx["X"] = res, X

    def s_shap():
        from .core import explain as ex
        from .core.models import REGISTRY
        r = ctx["res"].get("xgb")
        sh = ex.compute_shap(r.pipeline, REGISTRY["xgb"], ctx["X"].head(60), "classification")
        assert sh.values.shape[0] == 60
        ctx["shap"] = sh

    def s_plots():
        from .core import plots as pl
        res = ctx["res"]
        items = [(r.name, res.y_oof(r), r.oof[:, 1]) for r in res.results]
        f = pl.new_figure()
        pl.plot_roc(f, items)
        pl.fig_to_png(f)
        f = pl.new_figure()
        pl.plot_shap_summary(f, ctx["shap"])
        ctx["png"] = pl.fig_to_png(f)

    def s_cluster():
        from .core import cluster as cl
        from .core.models import REGISTRY
        ds = ctx["ds"]
        r = cl.run_clustering(ds.base(), ["age", "chol", "thalach"], ds.base_types(), REGISTRY["kmeans"], {"n_clusters": 3})
        assert r.n_clusters == 3

    def s_report():
        import pandas as pd
        from .core.report import Report
        d = Path(tempfile.mkdtemp())
        rep = Report("selftest")
        rep.add_heading("h")
        rep.add_table(pd.DataFrame({"a": [1.5, 2.5]}), "t")
        rep.add_figure(ctx["png"], "f")
        rep.add_ai("## ok\n- x\n\n| a | b |\n|---|---|\n| 1 | 2 |")
        rep.save_html(d / "r.html")
        rep.save_docx(d / "r.docx")
        rep.save_markdown(d / "r.md")
        rep.save_pdf(d / "r.pdf")
        assert (d / "r.pdf").read_bytes()[:5] == b"%PDF-"

    def s_io():
        from .core import io as mio
        from .core.sample_data import write_samples
        d = Path(tempfile.mkdtemp())
        paths = write_samples(d)
        for k in ("csv", "txt", "xlsx"):
            df, _ = mio.load_table(paths[k])
            assert df.shape == (303, 14)

    def s_gui():
        from .gui.main_window import MainWindow
        from .gui.theme import apply_theme
        apply_theme(app)
        w = MainWindow()
        assert len(w.pages) == 10
        w.pages["data"].load_sample()
        app.processEvents()

    for name, fn in (("读取 csv/txt/xlsx", s_io), ("数据集与预处理", s_data), ("训练 5 种分类模型(含 XGBoost/LightGBM)", s_models),
                     ("SHAP", s_shap), ("图表渲染", s_plots), ("聚类", s_cluster), ("报告导出 HTML/Word/Markdown/PDF", s_report),
                     ("界面构造(10 页)", s_gui)):
        step(name, fn)
    text = "\n".join(lines) + ("\nSELFTEST OK" if ok else "\nSELFTEST FAILED")
    print(text)
    try:
        Path("selftest_result.txt").write_text(text, encoding="utf-8")
    except OSError:
        pass
    return 0 if ok else 1


def main(argv=None) -> int:
    if "--selftest" in (argv if argv is not None else sys.argv[1:]):
        return selftest()
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

    from . import APP_NAME
    from .gui.main_window import MainWindow
    from .gui.theme import apply_theme

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    apply_theme(app)

    def hook(exc_type, exc, tb):  # 未捕获异常：弹窗而不是闪退
        import traceback
        traceback.print_exception(exc_type, exc, tb)
        QMessageBox.critical(None, "程序出错", f"{exc_type.__name__}: {exc}\n\n详细信息已输出到控制台。")

    sys.excepthook = hook
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
