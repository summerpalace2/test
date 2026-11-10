"""分析报告：收集“文字/表格/图表”条目，导出为 HTML / Markdown / Word / PDF。"""
from __future__ import annotations

import base64
import datetime as dt
import html
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import markdown as md
import pandas as pd

DISCLAIMER = "本报告由 MedMiner 自动生成，仅用于教学与研究，不构成临床诊断或治疗建议。"


def fmt_num(v) -> str:
    """表格数字：最多 3 位小数并去掉尾零（0.000 -> 0，54.145 保持）。"""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return ""
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.4g}"
    return str(v)


@dataclass
class ReportItem:
    kind: str  # heading | text | table | figure | ai
    title: str = ""
    content: Any = None  # text/ai: str；table: DataFrame；figure: PNG bytes
    source: str = ""  # 来源页面，用于列表显示

    def label(self) -> str:
        k = {"heading": "章节", "text": "文字", "table": "表格", "figure": "图表", "ai": "AI 解读"}[self.kind]
        return f"[{k}] {self.title or self._preview()}"

    def _preview(self) -> str:
        if isinstance(self.content, str):
            return re.sub(r"[*#>|`_]", "", self.content).strip().replace("\n", " ")[:30]
        return ""


@dataclass
class Report:
    title: str = "医学数据挖掘分析报告"
    author: str = ""
    items: list[ReportItem] = field(default_factory=list)

    def add(self, item: ReportItem):
        self.items.append(item)

    def add_heading(self, t): self.add(ReportItem("heading", t))
    def add_text(self, t, title=""): self.add(ReportItem("text", title, t))
    def add_table(self, df, title="", source=""): self.add(ReportItem("table", title, df.copy(), source))
    def add_figure(self, png: bytes, title="", source=""): self.add(ReportItem("figure", title, png, source))
    def add_ai(self, text, title="AI 分析", source="AI"): self.add(ReportItem("ai", title, text, source))

    def move(self, i: int, j: int):
        if 0 <= i < len(self.items) and 0 <= j < len(self.items):
            self.items.insert(j, self.items.pop(i))

    def remove(self, i: int):
        del self.items[i]

    # ---------------- HTML ----------------
    CSS = """
    body{font-family:'Microsoft YaHei','PingFang SC','Noto Sans CJK SC','WenQuanYi Zen Hei',sans-serif;max-width:900px;margin:32px auto;
    padding:0 20px;color:#1f2937;line-height:1.7}
    h1{font-size:26px;border-bottom:3px solid #2f6fde;padding-bottom:8px} h2{font-size:20px;margin-top:32px;color:#1d4ed8}
    h3{font-size:16px} .meta{color:#6b7280;font-size:13px}
    table{border-collapse:collapse;margin:12px 0;font-size:13px;width:100%} th,td{border:1px solid #d1d5db;padding:5px 8px;text-align:left}
    th{background:#eff6ff} tr:nth-child(even) td{background:#f9fafb}
    figure{margin:16px 0;text-align:center} figure img{max-width:100%} figcaption{color:#6b7280;font-size:13px}
    .ai{background:#f5f3ff;border-left:4px solid #8a5cc2;padding:6px 16px;margin:14px 0;border-radius:4px}
    .disclaimer{margin-top:40px;font-size:12px;color:#6b7280;border-top:1px solid #e5e7eb;padding-top:8px}
    """

    def to_html(self, for_qt: bool = False) -> str:
        parts = [f"<h1>{html.escape(self.title)}</h1>",
                 f"<p class='meta'>{html.escape(self.author + '  ' if self.author else '')}"
                 f"生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}</p>"]
        for it in self.items:
            if it.kind == "heading":
                parts.append(f"<h2>{html.escape(it.title)}</h2>")
            elif it.kind in ("text", "ai"):
                body = md.markdown(it.content or "", extensions=["tables", "fenced_code", "sane_lists"])
                if it.title and it.kind != "text":
                    body = f"<h3>{html.escape(it.title)}</h3>" + body
                parts.append(f"<div class='ai'>{body}</div>" if it.kind == "ai" else body)
            elif it.kind == "table":
                if it.title:
                    parts.append(f"<h3>{html.escape(it.title)}</h3>")
                parts.append(it.content.to_html(index=False, border=1 if for_qt else 0, na_rep="", float_format=fmt_num,
                                                escape=True))
            elif it.kind == "figure":
                b64 = base64.b64encode(it.content).decode()
                cap = f"<figcaption>{html.escape(it.title)}</figcaption>" if it.title else ""
                parts.append(f"<figure><img src='data:image/png;base64,{b64}'/>{cap}</figure>")
        parts.append(f"<p class='disclaimer'>{DISCLAIMER}</p>")
        css = self.CSS + (" td,th{white-space:nowrap;font-size:11px} table{width:auto}" if for_qt else "")
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(self.title)}</title>"
                f"<style>{css}</style></head><body>{''.join(parts)}</body></html>")

    def save_html(self, path):
        Path(path).write_text(self.to_html(), encoding="utf-8")

    # ---------------- Markdown ----------------
    def save_markdown(self, path):
        """Markdown + 同名 _images 目录。"""
        p = Path(path)
        img_dir = p.with_name(p.stem + "_images")
        lines = [f"# {self.title}", "", f"{self.author}  生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}", ""]
        n = 0
        for it in self.items:
            if it.kind == "heading":
                lines += [f"## {it.title}", ""]
            elif it.kind in ("text", "ai"):
                if it.title and it.kind != "text":
                    lines += [f"### {it.title}", ""]
                lines += [it.content or "", ""]
            elif it.kind == "table":
                if it.title:
                    lines += [f"### {it.title}", ""]
                lines += [_df_to_md(it.content), ""]
            elif it.kind == "figure":
                img_dir.mkdir(exist_ok=True)
                n += 1
                (img_dir / f"fig{n}.png").write_bytes(it.content)
                lines += [f"![{it.title}]({img_dir.name}/fig{n}.png)", ""]
        lines += ["---", DISCLAIMER]
        p.write_text("\n".join(lines), encoding="utf-8")

    # ---------------- Word ----------------
    def save_docx(self, path):
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.shared import Inches, Pt

        doc = Document()
        st = doc.styles["Normal"]
        st.font.name = "Microsoft YaHei"
        st.font.size = Pt(10.5)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        doc.add_heading(self.title, 0)
        doc.add_paragraph(f"{self.author}  生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}")
        for it in self.items:
            if it.kind == "heading":
                doc.add_heading(it.title, 1)
            elif it.kind in ("text", "ai"):
                if it.title and it.kind != "text":
                    doc.add_heading(it.title, 2)
                _md_to_docx(doc, it.content or "")
            elif it.kind == "table":
                if it.title:
                    doc.add_heading(it.title, 2)
                _df_to_docx(doc, it.content)
            elif it.kind == "figure":
                doc.add_picture(io.BytesIO(it.content), width=Inches(6.0))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                if it.title:
                    cap = doc.add_paragraph(it.title)
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = doc.add_paragraph(DISCLAIMER)
        p.runs[0].font.size = Pt(8.5)
        doc.save(str(path))

    # ---------------- PDF（需要 Qt） ----------------
    def save_pdf(self, path):
        from PySide6.QtCore import QMarginsF, QSizeF
        from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter, QTextDocument

        w = QPdfWriter(str(path))
        w.setPageSize(QPageSize(QPageSize.A4))
        w.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Millimeter)
        doc = QTextDocument()
        doc.setHtml(self.to_html(for_qt=True))
        doc.setPageSize(QSizeF(w.width(), w.height()))
        doc.print_(w)


def _df_to_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else fmt_num(v).replace("|", "/")
                                       for v in r) + " |")
    return "\n".join(lines)


def _df_to_docx(doc, df: pd.DataFrame):
    from docx.shared import Pt
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for _, r in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = "" if pd.isna(v) else fmt_num(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8.5)


_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _add_runs(par, text: str):
    pos = 0
    for m in _BOLD.finditer(text):
        par.add_run(text[pos:m.start()])
        par.add_run(m.group(1)).bold = True
        pos = m.end()
    par.add_run(text[pos:])


def _md_to_docx(doc, text: str):
    """把常见 Markdown（标题、列表、粗体、表格、引用）转成 Word 段落。"""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
            continue
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                if not re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            width = max(len(r) for r in rows)
            t = doc.add_table(rows=0, cols=width)
            t.style = "Light Grid Accent 1"
            for r in rows:
                cells = t.add_row().cells
                for k, v in enumerate(r):
                    cells[k].text = v
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", ln)
        if m:
            doc.add_heading(m.group(2), min(len(m.group(1)) + 1, 4))
        elif re.match(r"^\s*[-*]\s+", ln):
            _add_runs(doc.add_paragraph(style="List Bullet"), re.sub(r"^\s*[-*]\s+", "", ln))
        elif re.match(r"^\s*\d+[.)]\s+", ln):
            _add_runs(doc.add_paragraph(style="List Number"), re.sub(r"^\s*\d+[.)]\s+", "", ln))
        elif ln.startswith(">"):
            p = doc.add_paragraph()
            r = p.add_run(ln.lstrip("> ").strip())
            r.italic = True
        else:
            _add_runs(doc.add_paragraph(), ln)
        i += 1
