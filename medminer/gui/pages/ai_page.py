"""⑨ AI 分析：把各步结果的统计摘要交给大模型解读，支持流式输出、追问、发送前预览。"""
from __future__ import annotations

import html
import time

import pandas as pd
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter,
                               QVBoxLayout, QWidget)

from ...ai import context as actx
from ...ai import prompts
from ...ai.client import LLMError, stream_chat
from ...ai.config import LLMConfig, save_config
from ...core.medical import binary_confusion_metrics
from ..widgets.mdview import MarkdownView, md_to_html
from .base import Page, group, h1, hint, scroll
from .model_manager import ModelManagerDialog


class StreamWorker(QThread):
    chunk = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, cfg: LLMConfig, messages: list[dict], system: str, parent=None):
        super().__init__(parent)
        self.cfg, self.messages, self.system = cfg, messages, system
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            for piece in stream_chat(self.cfg, self.messages, self.system, cancel=lambda: self._cancel):
                self.chunk.emit(piece)
            self.done.emit()
        except LLMError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class PreviewDialog(QDialog):
    def __init__(self, system: str, user: str, tokens: int, cfg: LLMConfig, anon: actx.Anonymizer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("发送前预览")
        self.resize(780, 620)
        l = QVBoxLayout(self)
        dest = f"{cfg.preset} · {cfg.model} · {cfg.base_url}"
        l.addWidget(hint(f"将发送到：{dest}\n约 {tokens} tokens。以下是实际发出的全部内容（不含 API Key）。"
                         + ("列名已替换为代号：\n" + anon.legend().replace("\n", "；") if anon.enabled and anon.back else ""), warn=True))
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setPlainText(f"【系统提示】\n{system}\n\n【用户消息】\n{user}")
        l.addWidget(box, 1)
        bb = QDialogButtonBox()
        bb.addButton("发送", QDialogButtonBox.AcceptRole)
        bb.addButton("取消", QDialogButtonBox.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        l.addWidget(bb)


TASK_BUTTONS = [("profile", "解读数据画像"), ("preprocess", "预处理建议"), ("stats", "解读统计结果"), ("models", "解读模型结果"),
                ("cluster", "解读聚类结果"), ("patient", "解读单个样本"), ("report", "撰写报告结论")]


class AIPage(Page):
    key, title = "ai", "AI 分析"

    def __init__(self, state):
        super().__init__(state)
        self.history: list[dict] = []
        self.transcript: list[tuple[str, str]] = []  # (role, text)；assistant 的文本为原始（未还原列名）
        self.anon = actx.Anonymizer([], False)
        self.worker: StreamWorker | None = None
        self._streaming = ""
        self._last_render = 0.0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._render)

        lay = self.root_layout()
        top = QHBoxLayout()
        top.addWidget(h1("AI 分析"))
        top.addStretch(1)
        self.cfg_lbl = QLabel("")
        self.cfg_lbl.setObjectName("hint")
        top.addWidget(self.cfg_lbl)
        top.addWidget(QLabel("当前模型："))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(280)
        self.model_combo.setToolTip("在已保存的模型配置之间切换（每套配置有各自的 Base URL 和 Key）")
        self.model_combo.activated.connect(self._pick_profile)
        top.addWidget(self.model_combo)
        b = QPushButton("管理模型…")
        b.setObjectName("primary")
        b.clicked.connect(self._manage)
        top.addWidget(b)
        lay.addLayout(top)
        lay.addWidget(hint("大模型只会收到统计摘要（不含逐行患者数据）。回答仅供教学研究参考，请结合专业知识核对其中的数字与结论。"))

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)
        v = QVBoxLayout()
        self.task_btns: dict[str, QPushButton] = {}
        for k, text in TASK_BUTTONS:
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False, key=k: self._send_task(key))
            v.addWidget(btn)
            self.task_btns[k] = btn
        ll.addWidget(group("一键解读（基于当前结果）", v))
        self.avail = hint("")
        ll.addWidget(self.avail)
        v2 = QVBoxLayout()
        self.chk_ctx = QCheckBox("追问时附带数据与模型摘要")
        self.chk_ctx.setChecked(True)
        v2.addWidget(self.chk_ctx)
        bc = QPushButton("清空对话")
        bc.clicked.connect(self._clear)
        v2.addWidget(bc)
        ll.addWidget(group("对话", v2))
        ll.addStretch(1)
        split.addWidget(scroll(left, 230))

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.view = MarkdownView()
        rl.addWidget(self.view, 1)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("在此输入问题，例如：为什么随机森林不如逻辑回归？灵敏度该怎么取舍？（Ctrl+Enter 发送）")
        self.input.setMaximumHeight(84)
        rl.addWidget(self.input)
        row = QHBoxLayout()
        self.btn_send = QPushButton("发送")
        self.btn_send.setObjectName("primary")
        self.btn_send.clicked.connect(self._send_chat)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_add = QPushButton("把最后一条回答加入报告")
        self.btn_add.clicked.connect(self._to_report)
        row.addWidget(self.btn_send)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        row.addWidget(self.btn_add)
        rl.addLayout(row)
        split.addWidget(right)
        split.setSizes([230, 900])
        lay.addWidget(split, 1)

        state.eval_done.connect(self._update_avail)
        state.cluster_done.connect(self._update_avail)
        state.dataset_changed.connect(self._on_dataset)
        state.data_updated.connect(self._update_avail)
        state.config_changed.connect(self._update_cfg)
        self._update_cfg()
        self._update_avail()
        self._render()

    # ---------- 状态 ----------
    def _update_cfg(self):
        c = self.state.llm
        ok, why = c.is_ready()
        self.cfg_lbl.setText(f"隐藏列名：{'开' if c.anonymize else '关'}" + ("" if ok else f"　⚠ {why}"))
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for p in c.profiles:
            self.model_combo.addItem(f"{p['name']} · {p['model'] or '未选模型'}", p["name"])
        if c.active_name not in c.profile_names():  # 当前是没保存过的临时配置
            self.model_combo.insertItem(0, f"（临时）{c.preset} · {c.model or '未选模型'}", None)
        i = self.model_combo.findData(c.active_name or None)
        self.model_combo.setCurrentIndex(max(0, i))
        self.model_combo.blockSignals(False)

    def _pick_profile(self, index: int):
        name = self.model_combo.itemData(index)
        if name and self.state.llm.activate(name):
            try:
                save_config(self.state.llm)
            except OSError as e:
                self.error("配置未能保存", str(e))
            self.state.config_changed.emit()

    def _on_dataset(self):
        self._clear()
        self._update_avail()

    def _available(self) -> dict[str, tuple[bool, str]]:
        s = self.state
        has = s.ds is not None
        c = s.cache
        return {
            "profile": (has, "请先载入数据"),
            "preprocess": (has, "请先载入数据"),
            "stats": (has and ("table1" in c or "or" in c), "请先在“统计分析”生成基线表或 OR"),
            "models": (s.eval is not None, "请先在“建模评估”训练模型"),
            "cluster": (s.cluster is not None, "请先在“聚类分析”运行聚类"),
            "patient": ("patient" in c, "请先在“预测”页预测一个样本"),
            "report": (has, "请先载入数据"),
        }

    def _update_avail(self):
        av = self._available()
        miss = []
        for k, (ok, why) in av.items():
            self.task_btns[k].setEnabled(ok)
            self.task_btns[k].setToolTip("" if ok else why)
            if not ok:
                miss.append(f"“{dict(TASK_BUTTONS)[k]}”：{why}")
        self.avail.setText("灰色按钮需要先完成对应步骤。" if miss else "所有解读均可用。")

    # ---------- 上下文 ----------
    def _make_anon(self) -> actx.Anonymizer:
        ds = self.state.ds
        cols = []
        if ds is not None:
            cols = list(dict.fromkeys(list(ds.raw.columns) + list(ds.current().columns)))
        return actx.Anonymizer(cols, self.state.llm.anonymize)

    def _threshold_info(self, anon) -> str:
        b = self.state.eval
        if not b or b.task != "classification" or len(b.res.classes) != 2:
            return ""
        r = b.res.get(b.selected)
        y, p = b.res.y_oof(r), r.oof[:, 1]
        m = binary_confusion_metrics(y, (p >= b.threshold).astype(int))
        return (f"当前选中模型 {r.name}，在决策阈值 {b.threshold:.2f} 下（折外预测）：灵敏度 {m['sensitivity']:.3f}，特异度 {m['specificity']:.3f}，"
                f"PPV {m['ppv']:.3f}，NPV {m['npv']:.3f}，漏诊 FN={m['FN']}，误诊 FP={m['FP']}。")

    def build_context(self, task: str) -> str:
        s, cfg = self.state, self.state.llm
        anon = self.anon = self._make_anon()
        ds = s.ds
        parts: list[str] = []
        send_desc = cfg.send_descriptions and not anon.enabled

        def profile():
            return actx.dataset_profile(ds.current(), ds.current_types(), ds.descriptions, ds.target, anon, send_desc)

        def stats_part():
            out = []
            if "table1" in s.cache:
                out.append(actx.table_one_summary(s.cache["table1"].assign(变量=lambda d: d["变量"].map(anon.mask))))
            if "or" in s.cache:
                out.append(actx.or_summary(s.cache["or"], anon))
            return out

        def model_part():
            b = s.eval
            return actx.model_summary(b.res, anon, b.importance, "变量", b.selected, self._threshold_info(anon))

        def cluster_part():
            c = s.cluster
            return actx.cluster_summary(c.res, c.profile, anon, c.crosstab, c.truth_metrics)

        def patient_part():
            p = s.cache["patient"]
            contrib = p["contrib"] if p["contrib"] is not None else pd.DataFrame({"特征": [], "取值": [], "SHAP": []})
            return actx.patient_summary(p["values"], p["prob"], p["threshold"], contrib, anon, p["model"])

        if task == "profile":
            parts.append(profile())
        elif task == "preprocess":
            parts += [profile(), actx.preprocess_summary(ds.log(), anon)]
        elif task == "stats":
            parts += stats_part()
        elif task == "models":
            parts.append(model_part())
        elif task == "cluster":
            parts.append(cluster_part())
        elif task == "patient":
            parts.append(patient_part())
        elif task in ("report", "chat"):
            if ds is not None:
                parts += [profile()]
                if task == "report":
                    parts.append(actx.preprocess_summary(ds.log(), anon))
            if task == "report":
                parts += stats_part()
            if s.eval:
                parts.append(model_part())
            if s.cluster and task == "report":
                parts.append(cluster_part())
        if anon.enabled and parts:
            parts.append("（注：变量名已替换为代号 V1、V2…；回答中请沿用这些代号。）")
        return "\n\n".join(parts)

    # ---------- 发送 ----------
    def _ready(self) -> bool:
        ok, why = self.state.llm.is_ready()
        if not ok:
            if QMessageBox.question(self, "尚未配置大模型", f"{why}\n\n现在去设置？") == QMessageBox.Yes:
                self._manage()
            return False
        if self.worker and self.worker.isRunning():
            return False
        return True

    def _send_task(self, task: str, confirm: bool = True):
        if not self._ready():
            return
        ok, why = self._available()[task]
        if not ok:
            self.error("还不能解读", why)
            return
        ctx = self.build_context(task)  # 同时刷新 self.anon
        q = self.input.toPlainText().strip()
        user = prompts.build_user_message(task, ctx, self.anon.mask(q))  # 提问里的真实列名同样要脱敏
        label = f"【{prompts.TASKS[task][0]}】" + (q if q else "")
        self._dispatch(user, label, confirm)
        if q:
            self.input.clear()

    def _send_chat(self, confirm: bool = True):
        q = self.input.toPlainText().strip()
        if not q or not self._ready():
            return
        if not self.history and self.chk_ctx.isChecked():
            # 对话的第一条消息携带数据/模型摘要，之后的追问沿用上下文
            user = prompts.build_user_message("chat", self.build_context("chat"), self.anon.mask(q))
        else:
            if not self.history:
                self.anon = self._make_anon()
            user = self.anon.mask(q)
        self._dispatch(user, q, confirm)
        self.input.clear()

    def _dispatch(self, user: str, label: str, confirm: bool = True):
        cfg = self.state.llm
        msgs = (self.history + [{"role": "user", "content": user}])[-12:]
        if msgs and msgs[0]["role"] != "user":
            msgs = msgs[1:]
        if confirm and cfg.preview_before_send:
            dlg = PreviewDialog(prompts.SYSTEM, user, actx.estimate_tokens(prompts.SYSTEM + user), cfg, self.anon, self)
            if not dlg.exec():
                return
        self.history.append({"role": "user", "content": user})
        self.transcript.append(("user", label))
        self.transcript.append(("assistant", ""))
        self._streaming = ""
        self.btn_send.setEnabled(False)
        self.btn_stop.setEnabled(True)
        for b in self.task_btns.values():
            b.setEnabled(False)
        self.worker = StreamWorker(cfg, msgs, prompts.SYSTEM, self)
        self.worker.chunk.connect(self._on_chunk)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()
        self._render()

    def _on_chunk(self, piece: str):
        self._streaming += piece
        self.transcript[-1] = ("assistant", self._streaming)
        if time.time() - self._last_render > 0.08:
            self._render()
        elif not self._timer.isActive():
            self._timer.start(90)

    def _finish_ui(self):
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._update_avail()

    def _on_done(self):
        text = self._streaming.strip()
        self.history.append({"role": "assistant", "content": text})
        self.transcript[-1] = ("assistant", text)
        self._finish_ui()
        self._render()

    def _on_fail(self, msg: str):
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()  # 失败的这一轮不进入上下文，方便重试
        partial = self._streaming.strip()
        self.transcript[-1] = ("assistant", (partial + "\n\n" if partial else "") + f"> **出错**：{msg}")
        self._finish_ui()
        self._render()

    def _stop(self):
        if self.worker:
            self.worker.cancel()
        self.btn_stop.setEnabled(False)

    def _clear(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        self.history.clear()
        self.transcript.clear()
        self._streaming = ""
        self._finish_ui()
        self._render()

    # ---------- 显示 ----------
    def _render(self):
        self._last_render = time.time()
        if not self.transcript:
            self.view.set_markdown("### 开始分析\n左侧按钮会把当前结果的**统计摘要**发给大模型并给出解读；也可以直接在下方提问。\n\n"
                                   "- 首次使用请点右上角“管理模型…”，选服务商（Claude / OpenAI / 智谱 GLM / DeepSeek …）并填上你自己的 API Key，也可选本地 Ollama\n"
                                   "- 默认隐藏列名、每次发送前可预览内容，以保护隐私\n"
                                   "- 回答里的数字请与结果页核对")
            return
        parts = []
        for role, text in self.transcript:
            if role == "user":
                parts.append("<table width='100%' cellpadding='9' cellspacing='0'><tr><td bgcolor='#EAF1FF'>"
                             f"<b style='color:#2459B5'>你</b>&nbsp; {html.escape(text)}</td></tr></table>")
            else:
                shown = md_to_html(self.anon.restore(text)) if text else "<i style='color:#8A94A6'>思考中…</i>"
                parts.append(f"<p style='margin:10px 0 2px 0'><b style='color:#146356'>AI</b></p>{shown}")
        self.view.set_html_body("<br>".join(parts))

    # ---------- 其它 ----------
    def _manage(self):
        dlg = ModelManagerDialog(self.state.llm, self)
        if dlg.exec():
            self.state.llm = dlg.result_config()
            try:
                save_config(self.state.llm)
            except OSError as e:
                self.error("配置未能保存", str(e))
            self.state.config_changed.emit()

    def _to_report(self):
        last = next((t for r, t in reversed(self.transcript) if r == "assistant" and t.strip()), None)
        if not last or last.lstrip().startswith("> **出错**"):
            self.error("没有可加入的回答", "请先获得一条成功的 AI 回答。")
            return
        title = next((t for r, t in reversed(self.transcript) if r == "user"), "AI 分析")
        text = self.anon.restore(last)
        self.state.ai_notes.append((title, text))
        self.state.report_add_ai(text, title[:30])

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and e.modifiers() & Qt.ControlModifier and self.input.hasFocus():
            self._send_chat()
        else:
            super().keyPressEvent(e)
