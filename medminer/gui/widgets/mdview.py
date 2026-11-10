"""Markdown 渲染视图（AI 回答、帮助文本）。"""
from __future__ import annotations

import markdown as md
from PySide6.QtWidgets import QTextBrowser

CSS = """
body{font-size:13px;line-height:1.65;color:#1f2937}
h1,h2,h3{color:#1d4ed8;margin:10px 0 4px 0} h2{font-size:16px} h3{font-size:14px}
table{border-collapse:collapse;margin:6px 0} th,td{border:1px solid #d1d5db;padding:3px 8px} th{background:#eff6ff}
blockquote{color:#6b7280;margin:6px 0;padding-left:10px;border-left:3px solid #c7d2fe}
code{background:#f3f4f6;padding:1px 4px}
"""


def md_to_html(text: str) -> str:
    return f"<style>{CSS}</style>" + md.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "nl2br"])


class MarkdownView(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.document().setDefaultStyleSheet(CSS)

    def set_html_body(self, body: str):
        self.setHtml(f"<style>{CSS}</style>{body}")
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_markdown(self, text: str):
        self.setHtml(md_to_html(text))
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())
