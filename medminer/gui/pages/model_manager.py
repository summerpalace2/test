"""模型管理：多套大模型配置（服务商 + Base URL + 模型 + 自己的 Key），可测试连接、获取模型列表。"""
from __future__ import annotations

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMenu, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ...ai.client import LLMError, list_models, test_connection
from ...ai.config import PRESETS, PROTOCOL_LABELS, LLMConfig
from .base import group, hint


class PingWorker(QThread):
    ok = Signal(str)
    failed = Signal(str)

    def __init__(self, cfg: LLMConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        try:
            self.ok.emit(test_connection(self.cfg))
        except LLMError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class ListWorker(QThread):
    ok = Signal(list)
    failed = Signal(str)

    def __init__(self, cfg: LLMConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        try:
            self.ok.emit(list_models(self.cfg))
        except LLMError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


def _short(preset: str) -> str:
    """“Claude (Anthropic)” -> “Claude”，用作新配置的默认名称。"""
    return preset.split(" (")[0].split("（")[0].strip()


class ModelManagerDialog(QDialog):
    """左：已保存的配置列表；右：当前配置的表单。点“保存并使用”后返回新的 LLMConfig。"""

    def __init__(self, cfg: LLMConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型管理")
        self.resize(900, 640)
        self.base_cfg = cfg.clone()
        self.profiles: list[dict] = [dict(p) for p in cfg.profiles]
        if not self.profiles:  # 第一次使用：以当前配置作为第一套
            self.profiles = [cfg.current_as_profile(cfg.active_name or _short(cfg.preset))]
        self._row = -1
        self._loading = False
        self._workers: list[QThread] = []

        root = QVBoxLayout(self)
        root.addWidget(hint("每个服务商各存一套配置：Base URL、模型和你自己的 API Key。切换时不会互相覆盖。"
                            "只要服务商兼容 OpenAI 或 Anthropic 接口（包括各类中转/代理地址），选“自定义”填上 Base URL 就能用。"))
        body = QHBoxLayout()
        root.addLayout(body, 1)

        left = QVBoxLayout()
        self.listing = QListWidget()
        self.listing.setMinimumWidth(210)
        self.listing.currentRowChanged.connect(self._select)
        left.addWidget(self.listing, 1)
        row = QHBoxLayout()
        self.btn_add = QPushButton("＋ 新增")
        menu = QMenu(self.btn_add)
        for name in PRESETS:
            menu.addAction(name, lambda n=name: self._add(n))
        self.btn_add.setMenu(menu)
        self.btn_copy = QPushButton("复制")
        self.btn_copy.clicked.connect(self._copy)
        self.btn_del = QPushButton("删除")
        self.btn_del.setObjectName("danger")
        self.btn_del.clicked.connect(self._delete)
        for b in (self.btn_add, self.btn_copy, self.btn_del):
            row.addWidget(b)
        left.addLayout(row)
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(230)
        body.addWidget(lw)

        right = QVBoxLayout()
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        self.name_edit = QLineEdit()
        self.preset_box = QComboBox()
        self.preset_box.addItems(list(PRESETS))
        self.preset_box.activated.connect(self._preset_picked)
        self.proto_box = QComboBox()
        for k, v in PROTOCOL_LABELS.items():
            self.proto_box.addItem(v, k)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("例如 https://api.deepseek.com/v1（中转/代理地址也填这里）")
        self.model_box = QComboBox()
        self.model_box.setEditable(True)
        self.model_box.setMinimumWidth(260)
        self.btn_fetch = QPushButton("获取模型列表")
        self.btn_fetch.clicked.connect(self._fetch)
        mrow = QHBoxLayout()
        mrow.addWidget(self.model_box, 1)
        mrow.addWidget(self.btn_fetch)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("粘贴你的 API Key；本地模型（Ollama）可留空")
        self.show_key = QCheckBox("显示")
        self.show_key.toggled.connect(lambda on: self.key_edit.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        krow = QHBoxLayout()
        krow.addWidget(self.key_edit, 1)
        krow.addWidget(self.show_key)
        self.key_hint = QLabel("")
        self.key_hint.setOpenExternalLinks(True)
        self.key_hint.setWordWrap(True)
        self.key_hint.setTextFormat(Qt.RichText)
        self.key_hint.setMinimumHeight(44)  # 换行后的提示不被截断
        self.key_hint.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.key_hint.setObjectName("hint")
        form.addRow("配置名称", self.name_edit)
        form.addRow("服务商", self.preset_box)
        form.addRow("接口协议", self.proto_box)
        form.addRow("Base URL", self.url_edit)
        form.addRow("模型", mrow)
        form.addRow("API Key", krow)
        form.addRow("", self.key_hint)
        self.btn_test = QPushButton("测试连接")
        self.btn_test.clicked.connect(self._test)
        self.status = hint("")
        trow = QHBoxLayout()
        trow.addWidget(self.btn_test)
        trow.addWidget(self.status, 1)
        form.addRow("", trow)
        right.addWidget(group("连接", form))

        gen = QFormLayout()
        self.temp = QDoubleSpinBox()
        self.temp.setRange(0, 1.5)
        self.temp.setSingleStep(0.1)
        self.temp.setValue(cfg.temperature)
        self.maxtok = QSpinBox()
        self.maxtok.setRange(256, 32000)
        self.maxtok.setSingleStep(256)
        self.maxtok.setValue(cfg.max_tokens)
        self.timeout_box = QSpinBox()
        self.timeout_box.setRange(10, 600)
        self.timeout_box.setValue(int(cfg.timeout))
        grow = QHBoxLayout()
        for lab, w in (("温度", self.temp), ("最大输出 token", self.maxtok), ("超时(秒)", self.timeout_box)):
            grow.addWidget(QLabel(lab))
            grow.addWidget(w)
        grow.addStretch(1)
        gen.addRow(grow)
        right.addWidget(group("生成参数（所有配置共用）", gen))

        priv = QVBoxLayout()
        self.anon = QCheckBox("隐藏列名（发送 V1、V2… 代号，回答里自动还原）")
        self.anon.setChecked(cfg.anonymize)
        self.desc = QCheckBox("发送列的中文说明（隐藏列名时自动不发送）")
        self.desc.setChecked(cfg.send_descriptions)
        self.prev = QCheckBox("每次发送前先预览将要发送的内容")
        self.prev.setChecked(cfg.preview_before_send)
        for c in (self.anon, self.desc, self.prev):
            priv.addWidget(c)
        priv.addWidget(hint("只发送统计摘要，不发送逐行患者数据，文本列内容永不发送。Key 保存在本机用户目录 ~/.medminer/config.json，"
                            "不会写进项目文件或报告。用本地 Ollama 时数据完全不出本机。"))
        right.addWidget(group("隐私与安全", priv))
        right.addStretch(1)
        rw = QWidget()
        rw.setLayout(right)
        body.addWidget(rw, 1)

        foot = QHBoxLayout()
        foot.addStretch(1)
        ok = QPushButton("保存并使用选中的模型")
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        foot.addWidget(cancel)
        foot.addWidget(ok)
        root.addLayout(foot)

        self._refill_list()
        start = next((i for i, p in enumerate(self.profiles) if p["name"] == cfg.active_name), 0)
        self.listing.setCurrentRow(start)
        if self._row == -1:  # 列表加第一项时当前行可能已是 0，不会再触发信号，这里显式加载一次
            self._select(self.listing.currentRow())

    # ---------- 列表 ----------
    def _refill_list(self, select: int | None = None):
        self._loading = True
        self.listing.clear()
        for p in self.profiles:
            self.listing.addItem(f"{p['name']}\n{p['model'] or '（未选模型）'}")
        self._loading = False
        if select is not None and self.profiles:
            self.listing.setCurrentRow(select)

    def _unique(self, name: str) -> str:
        names = {p["name"] for p in self.profiles}
        if name not in names:
            return name
        i = 2
        while f"{name} ({i})" in names:
            i += 1
        return f"{name} ({i})"

    def _add(self, preset: str):
        self._commit()
        p = PRESETS[preset]
        self.profiles.append({"name": self._unique(_short(preset)), "preset": preset, "protocol": p["protocol"],
                              "base_url": p["base_url"], "model": p["model"], "api_key": ""})
        self._refill_list(len(self.profiles) - 1)

    def _copy(self):
        self._commit()
        if 0 <= self._row < len(self.profiles):
            p = dict(self.profiles[self._row])
            p["name"] = self._unique(p["name"] + " 副本")
            self.profiles.append(p)
            self._refill_list(len(self.profiles) - 1)

    def _delete(self):
        if 0 <= self._row < len(self.profiles):
            del self.profiles[self._row]
            self._row = -1
            self._refill_list(min(max(self.listing.currentRow(), 0), len(self.profiles) - 1) if self.profiles else None)
            if not self.profiles:
                self._load(None)

    def _select(self, row: int):
        if self._loading:
            return
        self._commit()
        self._row = row
        self._load(self.profiles[row] if 0 <= row < len(self.profiles) else None)

    # ---------- 表单 <-> 配置 ----------
    def _commit(self):
        """把表单内容写回当前选中的配置。"""
        if self._loading or not (0 <= self._row < len(self.profiles)):
            return
        p = self.profiles[self._row]
        name = self.name_edit.text().strip() or _short(self.preset_box.currentText())
        if name != p["name"] and name in {q["name"] for i, q in enumerate(self.profiles) if i != self._row}:
            name = self._unique(name)
        p.update(name=name, preset=self.preset_box.currentText(), protocol=self.proto_box.currentData(),
                 base_url=self.url_edit.text().strip(), model=self.model_box.currentText().strip(),
                 api_key=self.key_edit.text().strip())
        item = self.listing.item(self._row)
        if item is not None:
            item.setText(f"{p['name']}\n{p['model'] or '（未选模型）'}")

    def _load(self, p: dict | None):
        self._loading = True
        enabled = p is not None
        for w in (self.name_edit, self.preset_box, self.proto_box, self.url_edit, self.model_box, self.key_edit,
                  self.btn_fetch, self.btn_test, self.btn_copy, self.btn_del):
            w.setEnabled(enabled)
        if p is None:
            self.name_edit.clear()
            self.url_edit.clear()
            self.key_edit.clear()
            self.model_box.clear()
            self.key_hint.setText("还没有任何模型配置，点左下角“＋ 新增”选择服务商。")
        else:
            self.name_edit.setText(p["name"])
            self.preset_box.setCurrentText(p["preset"] if p["preset"] in PRESETS else "自定义 (OpenAI 兼容)")
            self.proto_box.setCurrentIndex(max(0, self.proto_box.findData(p["protocol"])))
            self.url_edit.setText(p["base_url"])
            self._fill_models(p["model"])
            self.key_edit.setText(p["api_key"])
            self._key_hint()
        self.status.setText("")
        self._loading = False

    def _fill_models(self, current: str, extra: list[str] | None = None):
        suggestions = list(PRESETS.get(self.preset_box.currentText(), {}).get("models", []))
        self.model_box.clear()
        self.model_box.addItems(list(dict.fromkeys((extra or []) + suggestions)))
        self.model_box.setCurrentText(current)

    def _key_hint(self):
        p = PRESETS.get(self.preset_box.currentText(), {})
        parts = []
        if p.get("key_url"):
            parts.append(f"获取 Key：<a href='{p['key_url']}'>{p['key_url']}</a>")
        if p.get("env"):
            parts.append(f"也可不在这里填，改用环境变量 {p['env']}")
        if not p.get("needs_key", True):
            parts.append("本地模型无需 Key；请先启动 Ollama")
        self.key_hint.setText("　".join(parts))

    def _preset_picked(self):
        """用户在下拉里换了服务商：带出该服务商的默认协议/地址/模型。Key 不动。"""
        name = self.preset_box.currentText()
        p = PRESETS[name]
        self.proto_box.setCurrentIndex(max(0, self.proto_box.findData(p["protocol"])))
        self.url_edit.setText(p["base_url"])
        self._fill_models(p["model"])
        self._key_hint()

    def _form_cfg(self) -> LLMConfig:
        c = self.base_cfg.clone()
        c.preset = self.preset_box.currentText()
        c.protocol = self.proto_box.currentData()
        c.base_url = self.url_edit.text().strip()
        c.model = self.model_box.currentText().strip()
        c.api_key = self.key_edit.text().strip()
        c.temperature, c.max_tokens, c.timeout = self.temp.value(), self.maxtok.value(), float(self.timeout_box.value())
        return c

    # ---------- 联网操作 ----------
    def _run(self, worker: QThread, btn: QPushButton, busy: str, on_ok, label_back: str):
        self.status.setText(busy)
        btn.setEnabled(False)
        self._workers.append(worker)

        def finish():
            btn.setEnabled(True)
            btn.setText(label_back)

        worker.ok.connect(lambda r: (finish(), on_ok(r)))
        worker.failed.connect(lambda m: (finish(), self.status.setText(f"失败：{m}")))
        worker.start()

    def _test(self):
        c = self._form_cfg()
        ok, why = c.is_ready()
        if not ok:
            self.status.setText(why)
            return
        self._run(PingWorker(c, self), self.btn_test, "正在测试连接…",
                  lambda r: self.status.setText(f"连接成功，模型回复：{r}"), "测试连接")

    def _fetch(self):
        c = self._form_cfg()
        cur = self.model_box.currentText()

        def got(ids: list):
            self._fill_models(cur if cur in ids else (cur or ids[0]), ids)
            self.status.setText(f"获取到 {len(ids)} 个模型，已放进“模型”下拉框（也可以手填）")
        self._run(ListWorker(c, self), self.btn_fetch, "正在获取模型列表…", got, "获取模型列表")

    # ---------- 结果 ----------
    def result_config(self) -> LLMConfig:
        self._commit()
        c = self.base_cfg.clone()
        c.profiles = [dict(p) for p in self.profiles]
        c.temperature, c.max_tokens, c.timeout = self.temp.value(), self.maxtok.value(), float(self.timeout_box.value())
        c.anonymize, c.send_descriptions, c.preview_before_send = self.anon.isChecked(), self.desc.isChecked(), self.prev.isChecked()
        if 0 <= self._row < len(self.profiles):
            c.apply_profile(self.profiles[self._row])
        elif not self.profiles:
            c.active_name = ""
        return c

    def done(self, code):  # 关闭前确保没有仍在运行的联网线程
        for w in self._workers:
            if w.isRunning():
                w.wait(1500)
        super().done(code)
