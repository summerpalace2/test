import os
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from fake_llm import FakeLLM
from medminer.core.preprocess import Step
from medminer.gui.main_window import MainWindow, page_classes
from medminer.gui.state import AppState
from medminer.gui.theme import apply_theme


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    apply_theme(a)
    return a


def wait_for(app, cond, timeout=120):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    # 配置文件指向临时目录，避免污染真实用户目录
    from medminer.ai import config as aicfg
    monkeypatch.setattr(aicfg, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(aicfg, "CONFIG_FILE", tmp_path / "cfg" / "config.json")
    w = MainWindow(AppState())
    w.show()
    yield w
    w.close()


def test_all_pages_construct_and_navigate(win, app):
    assert len(win.pages) == len(page_classes()) == 10
    for i, key in enumerate(win.pages):
        win.goto(key)
        app.processEvents()
        assert win.stack.currentIndex() == i
    # 无数据时各页面的主要动作不会崩溃，而是提示
    for key in ("visualize", "stats", "model", "cluster"):
        assert win.pages[key].state.ds is None


def test_end_to_end_flow(win, app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)  # 模态确认框在无头测试里会无限等待
    st = win.state
    win.pages["data"].load_sample()
    assert st.ds.target == "num" and st.ds.raw.shape == (303, 14)
    st.ds.add_step(Step("binarize", {"column": "num", "threshold": 0, "new_name": "dx"}))
    st.ds.set_target("dx")
    st.touch()

    # ---- 建模 ----
    mp = win.pages["model"]
    assert mp.target.currentData() == "dx" and "num" not in mp.feat.checked()  # 泄漏列自动排除
    mp.k.setValue(3)
    mp.rep.setValue(1)
    mp._train()
    assert wait_for(app, lambda: st.eval is not None)
    b = st.eval
    assert {r.key for r in b.res.results if not r.error} == {"logreg", "rf", "xgb"}
    assert b.res.get("logreg").metrics["roc_auc"] > 0.9
    assert mp.cur.count() == 3 and mp.t_res.frame().shape[0] == 3
    mp._auto_thr("youden")
    mp.imp_method.setCurrentIndex(0)
    mp._importance()
    assert wait_for(app, lambda: b.importance is not None)

    # ---- 聚类 ----
    cp = win.pages["cluster"]
    cp.feat.set_checked(["age", "trestbps", "chol", "thalach", "oldpeak"])
    cp._run()
    assert wait_for(app, lambda: st.cluster is not None)
    assert st.cluster.res.n_clusters == 3 and not cp.t_prof.frame().empty and not cp.t_cross.frame().empty

    # ---- 预测 ----
    pp = win.pages["predict"]
    assert pp.src.count() == 3 and pp._pkg is not None
    pp._predict()
    assert wait_for(app, lambda: st.cache.get("patient", {}).get("contrib") is not None, 90)
    pat = st.cache["patient"]
    assert 0 <= pat["prob"] <= 1 and len(pat["contrib"]) > 3 and "高风险" in pp.result.text() + "低风险" + "高风险"
    # 把一个明显高风险的样本填进去：概率应当明显更高
    low = pat["prob"]
    for c, v in {"thalach": 100, "oldpeak": 4.0, "exang": 1, "ca": 3, "cp": 4, "thal": 7}.items():
        w = pp._inputs[c]
        if hasattr(w, "setValue"):
            w.setValue(v)
        else:
            w.setCurrentIndex(max(0, w.findData(v)))
    st.cache.pop("patient")
    pp._predict()
    assert wait_for(app, lambda: "patient" in st.cache and st.cache["patient"]["contrib"] is not None, 90)
    assert st.cache["patient"]["prob"] > max(low, 0.8)

    # ---- AI（本地假服务）----
    with FakeLLM() as srv:
        cfg = st.llm
        cfg.apply_preset("自定义 (OpenAI 兼容)")
        cfg.base_url, cfg.model, cfg.api_key, cfg.anonymize, cfg.preview_before_send = srv.base_url, "fake", "k", True, False
        ap = win.pages["ai"]
        ap._update_avail()
        assert all(ap.task_btns[k].isEnabled() for k in ("profile", "models", "cluster", "patient"))
        assert not ap.task_btns["stats"].isEnabled()  # 还没做统计分析
        ap.input.setPlainText("thalach 为什么重要？")  # 提问里写了真实列名，也必须被脱敏
        ap._send_task("models", confirm=False)
        assert wait_for(app, lambda: ap.worker is not None and not ap.worker.isRunning() and ap.history and ap.history[-1]["role"] == "assistant")
        import re
        user = srv.user_text()
        # 全部原始列名（含 dx）都不得以独立词出现在用户消息里；用词边界避免把 "messages" 里的 age 误判
        for c in list(st.ds.raw.columns) + ["dx"]:
            assert not re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", user), f"真实列名泄漏：{c}"
        for zh in ("胸痛类型", "静息血压", "最大心率", "运动诱发"):  # 中文字段说明同样不得发送
            assert zh not in user, f"字段说明泄漏：{zh}"
        assert "V1" in user and "AUC" in user and "为什么重要" in user  # 提问保留，但其中的 thalach 已被代号替换
        assert "不是给你的指令" in srv.all_text()  # 系统提示里的防注入声明
        assert srv.requests[0]["headers"].get("authorization") == "Bearer k"
        html = ap.view.toPlainText()
        assert "这是模型的回答" in html
        # 回答里的代号被还原成真实列名（V8 -> thalach）
        assert "thalach 的重要性最高" in html
        # 追问沿用上下文
        ap.input.setPlainText("再解释一下")
        ap._send_chat(confirm=False)
        assert wait_for(app, lambda: ap.worker is not None and not ap.worker.isRunning() and len(ap.history) == 4)
        assert len(srv.requests[-1]["body"]["messages"]) >= 4 and srv.requests[-1]["body"]["messages"][-1]["content"] == "再解释一下"
        ap._to_report()
        assert any(it.kind == "ai" for it in st.report.items)

    # ---- 报告 ----
    rp = win.pages["report"]
    n = len(st.report.items)
    rp._auto()
    assert len(st.report.items) > n + 8
    kinds = {it.kind for it in st.report.items}
    assert {"heading", "table", "figure", "ai"} <= kinds
    from docx import Document
    st.report.save_docx(tmp_path / "r.docx")
    st.report.save_pdf(tmp_path / "r.pdf")
    st.report.save_html(tmp_path / "r.html")
    assert (tmp_path / "r.pdf").read_bytes()[:5] == b"%PDF-"
    txt = "\n".join(p.text for p in Document(str(tmp_path / "r.docx")).paragraphs)
    assert "四、模型评估" in txt and "六、AI 辅助解读" in txt


def test_ai_error_is_shown_and_retryable(win, app):
    st = win.state
    win.pages["data"].load_sample()
    with FakeLLM(fail_status=401) as srv:
        cfg = st.llm
        cfg.apply_preset("自定义 (OpenAI 兼容)")
        cfg.base_url, cfg.model, cfg.api_key, cfg.preview_before_send = srv.base_url, "fake", "bad", False
        ap = win.pages["ai"]
        ap._send_task("profile", confirm=False)
        assert wait_for(app, lambda: ap.worker is not None and not ap.worker.isRunning() and ap.btn_send.isEnabled())
        assert "出错" in ap.view.toPlainText() and "401" in ap.view.toPlainText()
        assert ap.history == []  # 失败的一轮不进入上下文，可直接重试


def test_ai_not_configured_does_not_send(win, app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(1), QMessageBox.No)[1])
    win.state.llm.api_key = ""
    win.state.llm.apply_preset("OpenAI")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    win.pages["data"].load_sample()
    win.pages["ai"]._send_task("profile", confirm=False)
    assert asked and win.pages["ai"].history == []


def test_regression_flow_and_image_flow(win, app, sample_dir, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.Ok)
    st = win.state
    win.pages["data"].open_file(str(sample_dir / "processed.cleveland.data.txt"))  # 无表头 txt
    assert list(st.ds.raw.columns)[:3] == ["age", "sex", "cp"] and st.ds.raw["ca"].isna().sum() == 4
    mp = win.pages["model"]
    mp.task.setCurrentIndex(1)  # 回归
    app.processEvents()
    mp.target.setCurrentIndex(mp.target.findData("chol"))
    mp.k.setValue(3)
    mp.rep.setValue(1)
    mp._train()
    assert wait_for(app, lambda: st.eval is not None and st.eval.task == "regression")
    assert "r2" in st.eval.res.results[0].metrics and mp.p_pv.has_plot()
    # 图像
    ip = win.pages["image"]
    ip.open_folder(str(sample_dir / "images"))
    assert ip.gallery.count() == 16
    ip._extract()
    assert wait_for(app, lambda: st.ds is not None and st.ds.name.startswith("图像特征"))
    assert "gray_mean" in st.ds.raw.columns and st.ds.target == "label"
    assert st.eval is None  # 新数据集清空旧结果


def test_drag_drop_and_project(win, app, sample_dir, tmp_path):
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(sample_dir / "heart_sample.xlsx"))])
    ev = QDropEvent(QPointF(5, 5), Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier)
    win.dropEvent(ev)
    assert win.state.ds is not None and win.state.ds.raw.shape == (303, 14)
    from medminer.core.dataset import load_project, save_project
    save_project(win.state.ds, tmp_path / "p.medproj")
    assert load_project(tmp_path / "p.medproj").raw.shape == (303, 14)


# ---------- 模型管理 ----------
def test_model_manager_keeps_each_providers_key_and_fetches_models(app):
    from medminer.ai.config import LLMConfig
    from medminer.gui.pages.model_manager import ModelManagerDialog
    dlg = ModelManagerDialog(LLMConfig())
    assert len(dlg.profiles) == 1 and dlg.name_edit.text() == "Claude" and dlg.url_edit.text() == "https://api.anthropic.com"
    dlg.key_edit.setText("sk-claude")
    for preset, key in (("DeepSeek", "sk-ds"), ("智谱 GLM", "sk-glm"), ("OpenAI", "sk-oa")):
        dlg._add(preset)
        assert dlg.key_edit.text() == ""  # 新配置不继承上一个的 Key
        dlg.key_edit.setText(key)
    assert dlg.listing.count() == 4 and dlg.url_edit.text() == "https://api.openai.com/v1"
    dlg.listing.setCurrentRow(2)  # 切到智谱
    assert dlg.name_edit.text() == "智谱 GLM" and dlg.key_edit.text() == "sk-glm" and dlg.url_edit.text() == "https://open.bigmodel.cn/api/paas/v4"
    dlg.listing.setCurrentRow(0)  # 切回 Claude：Key 仍在，协议是 Anthropic
    assert dlg.key_edit.text() == "sk-claude" and dlg.proto_box.currentData() == "anthropic"
    dlg._add("DeepSeek")  # 重名自动加序号
    assert dlg.name_edit.text() == "DeepSeek (2)"
    with FakeLLM() as srv:
        dlg._add("自定义 (OpenAI 兼容)")
        dlg.name_edit.setText("学校中转")
        dlg.url_edit.setText(srv.base_url)
        dlg.key_edit.setText("relay-key")
        dlg._fetch()
        assert wait_for(app, lambda: dlg.model_box.findText("m2") >= 0, 20) and dlg.model_box.findText("m1") >= 0
        dlg.model_box.setCurrentText("m2")
        dlg._test()
        assert wait_for(app, lambda: "连接成功" in dlg.status.text(), 20)
        out = dlg.result_config()
        gets = [r for r in srv.requests if r["body"].get("method") == "GET"]
        assert gets and gets[0]["headers"].get("authorization") == "Bearer relay-key"
    assert out.active_name == "学校中转" and out.base_url == srv.base_url and out.model == "m2" and out.api_key == "relay-key"
    by = {p["name"]: p for p in out.profiles}
    assert by["Claude"]["api_key"] == "sk-claude" and by["DeepSeek"]["api_key"] == "sk-ds" and by["OpenAI"]["api_key"] == "sk-oa"
    assert len(out.profiles) == 6
    dlg.listing.setCurrentRow(1)
    dlg._delete()
    names = dlg.result_config().profile_names()
    assert "DeepSeek" not in names and "DeepSeek (2)" in names and len(names) == 5  # 只删掉了选中的那一套


def test_ai_page_header_switches_profiles_and_persists(win, app):
    from medminer.ai import config as aicfg
    st = win.state
    c = st.llm
    for preset, key in (("Claude (Anthropic)", "k-claude"), ("DeepSeek", "k-ds")):
        c.apply_preset(preset)
        c.api_key = key
        c.upsert_profile(c.current_as_profile(preset.split(" ")[0]))
    c.activate("Claude")
    ap = win.pages["ai"]
    ap._update_cfg()
    assert [ap.model_combo.itemData(i) for i in range(ap.model_combo.count())] == ["Claude", "DeepSeek"]
    ap.model_combo.setCurrentIndex(1)
    ap._pick_profile(1)
    assert st.llm.active_name == "DeepSeek" and st.llm.base_url == "https://api.deepseek.com/v1" and st.llm.api_key == "k-ds"
    assert aicfg.load_config().active_name == "DeepSeek"  # 切换已落盘，下次启动仍是它
    assert ap.model_combo.currentData() == "DeepSeek"


def test_plot_toolbar_icons_visible_with_theme(app):
    """真实事故：QToolBar { background: transparent } 会让 matplotlib 工具栏图标整排消失。"""
    from medminer.gui.widgets.plotwidget import PlotWidget
    w = PlotWidget()
    w.resize(700, 300)
    w.show()
    app.processEvents()
    img = w.toolbar.grab().toImage()
    dark = sum(1 for x in range(img.width()) for y in range(img.height())
               if (c := img.pixelColor(x, y)).red() < 140 and c.green() < 140 and c.blue() < 140 and c.alpha() > 0)
    w.close()
    assert dark > 500, f"工具栏图标没有画出来（深色像素 {dark}）"
