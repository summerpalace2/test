"""大模型配置：提供商预设、持久化（~/.medminer/config.json，权限 600）、环境变量回退。"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("MEDMINER_HOME", Path.home() / ".medminer"))
CONFIG_FILE = CONFIG_DIR / "config.json"

# 名称 -> (协议, base_url, 默认模型, 候选模型, 环境变量名)
PRESETS: dict[str, dict] = {
    "Claude (Anthropic)": dict(protocol="anthropic", base_url="https://api.anthropic.com", model="claude-sonnet-5-5",
                               models=["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5", "claude-fable-5-1"],
                               env="ANTHROPIC_API_KEY"),
    "OpenAI": dict(protocol="openai", base_url="https://api.openai.com/v1", model="gpt-4o-mini",
                   models=["gpt-4o-mini", "gpt-4o", "gpt-4.1"], env="OPENAI_API_KEY"),
    "DeepSeek": dict(protocol="openai", base_url="https://api.deepseek.com/v1", model="deepseek-chat",
                     models=["deepseek-chat", "deepseek-reasoner"], env="DEEPSEEK_API_KEY"),
    "通义千问 (阿里云百炼)": dict(protocol="openai", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                          model="qwen-plus", models=["qwen-plus", "qwen-max", "qwen-turbo"], env="DASHSCOPE_API_KEY"),
    "智谱 GLM": dict(protocol="openai", base_url="https://open.bigmodel.cn/api/paas/v4", model="glm-4-flash",
                   models=["glm-4-flash", "glm-4-plus", "glm-4-air"], env="ZHIPUAI_API_KEY"),
    "Moonshot (Kimi)": dict(protocol="openai", base_url="https://api.moonshot.cn/v1", model="moonshot-v1-8k",
                            models=["moonshot-v1-8k", "moonshot-v1-32k"], env="MOONSHOT_API_KEY"),
    "本地 Ollama（数据不出本机）": dict(protocol="openai", base_url="http://localhost:11434/v1", model="qwen2.5:7b",
                              models=["qwen2.5:7b", "llama3.1:8b", "deepseek-r1:7b"], env="", needs_key=False),
    "自定义 (OpenAI 兼容)": dict(protocol="openai", base_url="", model="", models=[], env=""),
}
DEFAULT_PRESET = "Claude (Anthropic)"


@dataclass
class LLMConfig:
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
            return False, "尚未配置 API Key（在“AI 分析 → 设置”中填写）"
        return True, ""


def load_config() -> LLMConfig:
    cfg = LLMConfig()
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg: LLMConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_FILE, 0o600)  # Windows 上无效，忽略
    except OSError:
        pass
