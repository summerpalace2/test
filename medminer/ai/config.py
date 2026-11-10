"""大模型配置：服务商预设、多套模型配置（profiles）、持久化（~/.medminer/config.json，权限 600）、环境变量回退。"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("MEDMINER_HOME", Path.home() / ".medminer"))
CONFIG_FILE = CONFIG_DIR / "config.json"

# 名称 -> 协议 / 接口地址 / 默认模型 / 候选模型 / 环境变量名 / 获取 Key 的页面
# 候选模型只是下拉里的建议，界面里可以直接手填，或点“获取模型列表”从服务商拉取真实可用的模型。
PRESETS: dict[str, dict] = {
    "Claude (Anthropic)": dict(protocol="anthropic", base_url="https://api.anthropic.com", model="claude-sonnet-5-5",
                               models=["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5", "claude-fable-5-1"],
                               env="ANTHROPIC_API_KEY", key_url="https://console.anthropic.com/settings/keys"),
    "OpenAI": dict(protocol="openai", base_url="https://api.openai.com/v1", model="gpt-4o-mini",
                   models=["gpt-4o-mini", "gpt-4o", "gpt-4.1"], env="OPENAI_API_KEY",
                   key_url="https://platform.openai.com/api-keys"),
    "智谱 GLM": dict(protocol="openai", base_url="https://open.bigmodel.cn/api/paas/v4", model="glm-4-flash",
                   models=["glm-4-flash", "glm-4-plus", "glm-4-air"], env="ZHIPUAI_API_KEY",
                   key_url="https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys"),
    "DeepSeek": dict(protocol="openai", base_url="https://api.deepseek.com/v1", model="deepseek-chat",
                     models=["deepseek-chat", "deepseek-reasoner"], env="DEEPSEEK_API_KEY",
                     key_url="https://platform.deepseek.com/api_keys"),
    "通义千问 (阿里云百炼)": dict(protocol="openai", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                          model="qwen-plus", models=["qwen-plus", "qwen-max", "qwen-turbo"], env="DASHSCOPE_API_KEY",
                          key_url="https://bailian.console.aliyun.com/?tab=model#/api-key"),
    "Moonshot (Kimi)": dict(protocol="openai", base_url="https://api.moonshot.cn/v1", model="moonshot-v1-8k",
                            models=["moonshot-v1-8k", "moonshot-v1-32k"], env="MOONSHOT_API_KEY",
                            key_url="https://platform.moonshot.cn/console/api-keys"),
    "Google Gemini (OpenAI 兼容)": dict(protocol="openai", base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                                     model="gemini-2.5-flash", models=["gemini-2.5-flash", "gemini-2.5-pro"],
                                     env="GEMINI_API_KEY", key_url="https://aistudio.google.com/apikey"),
    "硅基流动 SiliconFlow": dict(protocol="openai", base_url="https://api.siliconflow.cn/v1",
                             model="Qwen/Qwen2.5-7B-Instruct", models=["Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-V3"],
                             env="SILICONFLOW_API_KEY", key_url="https://cloud.siliconflow.cn/account/ak"),
    "OpenRouter": dict(protocol="openai", base_url="https://openrouter.ai/api/v1", model="openai/gpt-4o-mini",
                       models=["openai/gpt-4o-mini", "anthropic/claude-sonnet-4.5"], env="OPENROUTER_API_KEY",
                       key_url="https://openrouter.ai/keys"),
    "本地 Ollama（数据不出本机）": dict(protocol="openai", base_url="http://localhost:11434/v1", model="qwen2.5:7b",
                              models=["qwen2.5:7b", "llama3.1:8b", "deepseek-r1:7b"], env="", needs_key=False,
                              key_url=""),
    "自定义 (OpenAI 兼容)": dict(protocol="openai", base_url="", model="", models=[], env="", key_url=""),
    "自定义 (Anthropic 兼容)": dict(protocol="anthropic", base_url="", model="", models=[], env="", key_url=""),
}
DEFAULT_PRESET = "Claude (Anthropic)"
PROTOCOL_LABELS = {"openai": "OpenAI 兼容（/chat/completions）", "anthropic": "Anthropic（/v1/messages）"}

_PROFILE_KEYS = ("name", "preset", "protocol", "base_url", "model", "api_key")


@dataclass
class LLMConfig:
    """当前生效的配置（preset/protocol/base_url/model/api_key）+ 已保存的多套配置 profiles。
    其它代码只读写前面这些“当前生效”字段；切换配置 = 把某个 profile 的字段拷贝进来。"""
    preset: str = DEFAULT_PRESET
    protocol: str = "anthropic"  # anthropic | openai
    base_url: str = "https://api.anthropic.com"
    model: str = "claude-sonnet-5-5"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: float = 120.0
    anonymize: bool = True  # 隐藏列名（V1、V2…），回答里自动还原
    send_descriptions: bool = True  # 是否把列的中文说明发给模型
    preview_before_send: bool = True  # 每次发送前先预览将要发出的内容
    active_name: str = ""  # 当前使用的 profile 名称（空 = 未保存的临时配置）
    profiles: list[dict] = field(default_factory=list)

    # ---------- 生效配置 ----------
    def effective_key(self) -> str:
        if self.api_key:
            return self.api_key
        env = PRESETS.get(self.preset, {}).get("env", "")
        return os.environ.get(env, "") if env else os.environ.get("MEDMINER_API_KEY", "")

    def needs_key(self) -> bool:
        return PRESETS.get(self.preset, {}).get("needs_key", True)

    def apply_preset(self, name: str):
        p = PRESETS[name]
        self.preset, self.protocol, self.base_url, self.model = name, p["protocol"], p["base_url"], p["model"]

    def is_ready(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "请先填写接口地址（Base URL）"
        if not self.model:
            return False, "请先填写模型名称"
        if self.needs_key() and not self.effective_key():
            return False, "尚未配置 API Key（在“管理模型…”中填写）"
        return True, ""

    # ---------- 多套配置 ----------
    def current_as_profile(self, name: str | None = None) -> dict:
        return {"name": name or self.active_name or self.preset, "preset": self.preset, "protocol": self.protocol,
                "base_url": self.base_url, "model": self.model, "api_key": self.api_key}

    def profile_names(self) -> list[str]:
        return [p["name"] for p in self.profiles]

    def get_profile(self, name: str) -> dict | None:
        return next((p for p in self.profiles if p["name"] == name), None)

    def apply_profile(self, p: dict):
        """把某套配置设为当前生效（不改动 profiles 列表本身）。"""
        self.preset = p.get("preset", "自定义 (OpenAI 兼容)")
        self.protocol = p.get("protocol") or PRESETS.get(self.preset, {}).get("protocol", "openai")
        self.base_url, self.model, self.api_key = p.get("base_url", ""), p.get("model", ""), p.get("api_key", "")
        self.active_name = p["name"]

    def activate(self, name: str) -> bool:
        p = self.get_profile(name)
        if p is None:
            return False
        self.apply_profile(p)
        return True

    def upsert_profile(self, p: dict):
        p = {k: p.get(k, "") for k in _PROFILE_KEYS}
        for i, old in enumerate(self.profiles):
            if old["name"] == p["name"]:
                self.profiles[i] = p
                return
        self.profiles.append(p)

    def delete_profile(self, name: str):
        self.profiles = [p for p in self.profiles if p["name"] != name]
        if self.active_name == name:
            self.active_name = ""

    def display_name(self) -> str:
        return f"{self.active_name or self.preset} · {self.model or '未选模型'}"

    def clone(self) -> "LLMConfig":
        return LLMConfig(**copy.deepcopy(asdict(self)))


def load_config() -> LLMConfig:
    cfg = LLMConfig()
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    except (OSError, ValueError):
        pass
    # 旧版本只有单套配置：自动迁移成一个 profile，用户之前填的 Key 不会丢
    if not cfg.profiles and (cfg.api_key or cfg.preset != DEFAULT_PRESET):
        name = cfg.preset
        cfg.profiles = [cfg.current_as_profile(name)]
        cfg.active_name = name
    return cfg


def save_config(cfg: LLMConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_FILE, 0o600)  # Windows 上无效，忽略
    except OSError:
        pass
