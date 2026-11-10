import numpy as np
import pandas as pd
import pytest
from PIL import Image

from medminer.core import images as im
from medminer.core import plots as pl
from medminer.core.dataset import Dataset
from medminer.core.report import DISCLAIMER, Report


# ---------------- 图像 ----------------
def test_scan_and_features(sample_dir, tmp_path):
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "broken.png").write_bytes(b"not an image")
    info = im.scan_image_folder(sample_dir / "images")
    assert len(info) == 16 and set(info["label"]) == {"normal", "lesion"}
    assert (info["width"] == 96).all()
    assert len(im.scan_image_folder(tmp_path)) == 0  # 损坏图片被跳过而不是崩溃
    feats = im.extract_features(info, size=64)
    assert set(im.FEATURE_NAMES) <= set(feats.columns) and feats["gray_mean"].notna().all()
    # 病灶图像应当更亮、中心比边缘更亮：这是特征提取是否“有信息”的实质检验
    g = feats.groupby("label")[["gray_max", "center_border_diff", "entropy"]].mean()
    assert g.loc["lesion", "gray_max"] > g.loc["normal", "gray_max"]
    assert g.loc["lesion", "center_border_diff"] != g.loc["normal", "center_border_diff"]
    ds = Dataset.from_frame("img", feats)
    assert ds.types["gray_mean"] == "numeric" and ds.types["label"] == "binary"


def test_color_and_rgb_modes(tmp_path):
    arr = np.zeros((40, 40, 3), "uint8")
    arr[..., 0] = 200
    Image.fromarray(arr, "RGB").save(tmp_path / "red.png")
    Image.new("L", (30, 50), 128).save(tmp_path / "gray.jpg")
    info = im.scan_image_folder(tmp_path)
    f = im.extract_features(info, 32)
    red = f[f["file"] == "red.png"].iloc[0]
    assert red["r_mean"] > 190 and red["g_mean"] < 5
    assert "label" not in f.columns  # 没有子文件夹 => 无标签列


# ---------------- 报告 ----------------
@pytest.fixture
def report():
    fig = pl.new_figure()
    pl.draw_chart(fig, "histogram", pd.DataFrame({"a": np.random.default_rng(0).normal(size=100)}), {"a": "numeric"}, x="a")
    r = Report(title="心脏病分析", author="第3组")
    r.add_heading("一、数据概况")
    r.add_text("共 **303** 例样本。")
    r.add_table(pd.DataFrame({"变量": ["age", "chol"], "均值": [54.1234, 246.5]}), "描述统计")
    r.add_figure(pl.fig_to_png(fig, 80), "年龄分布")
    r.add_ai("## 结论\n- 模型 **AUC=0.91**\n1. 注意小样本\n\n| 模型 | AUC |\n|---|---|\n| LR | 0.91 |\n\n> 仅供教学", "模型解读")
    return r


def test_html(report, tmp_path):
    h = report.to_html()
    assert "<h1>心脏病分析</h1>" in h and "data:image/png;base64," in h and DISCLAIMER in h
    assert "<strong>303</strong>" in h and "<td>54.123</td>" in h and "class='ai'" in h and "<table>" in h
    p = tmp_path / "r.html"
    report.save_html(p)
    assert p.stat().st_size > 5000


def test_html_escapes_untrusted_content():
    r = Report(title="<script>alert(1)</script>")
    r.add_table(pd.DataFrame({"x": ["<img src=x onerror=alert(1)>"]}), "t<b>")
    h = r.to_html()
    assert "<script>alert(1)</script>" not in h and "<img src=x" not in h


def test_markdown_export(report, tmp_path):
    p = tmp_path / "r.md"
    report.save_markdown(p)
    text = p.read_text(encoding="utf-8")
    assert "# 心脏病分析" in text and "| 变量 | 均值 |" in text and "![年龄分布](r_images/fig1.png)" in text
    assert (tmp_path / "r_images" / "fig1.png").exists()


def test_docx_roundtrip(report, tmp_path):
    from docx import Document
    p = tmp_path / "r.docx"
    report.save_docx(p)
    d = Document(str(p))
    texts = [x.text for x in d.paragraphs]
    assert "心脏病分析" in texts and "一、数据概况" in texts and DISCLAIMER in texts
    assert any(x.style.name == "List Bullet" and "AUC=0.91" in x.text for x in d.paragraphs)
    assert len(d.tables) == 2  # 描述统计表 + AI 回答里的 markdown 表
    assert d.tables[1].rows[1].cells[1].text == "0.91"
    assert len(d.inline_shapes) == 1
    bold = [r.text for p_ in d.paragraphs for r in p_.runs if r.bold]
    assert "303" in bold and "AUC=0.91" in bold


def test_pdf_export(report, tmp_path):
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication.instance() or QGuiApplication([])
    p = tmp_path / "r.pdf"
    report.save_pdf(p)
    data = p.read_bytes()
    assert data[:5] == b"%PDF-" and len(data) > 5000


def test_reorder_remove(report):
    n = len(report.items)
    first = report.items[0]
    report.move(0, 2)
    assert report.items[2] is first
    report.remove(2)
    assert len(report.items) == n - 1 and first not in report.items
    assert report.items[0].label().startswith("[")
