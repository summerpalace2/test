"""大模型客户端：Anthropic Messages 与 OpenAI 兼容 Chat Completions，均支持流式输出。"""
from __future__ import annotations

import json
from typing import Callable, Iterator

import httpx

from .config import LLMConfig


class LLMError(Exception):
    """面向用户的错误信息（已本地化）。"""


def _friendly(status: int, body: str, cfg: LLMConfig) -> str:
    detail = body.strip()
    try:
        j = json.loads(body)
        detail = (j.get("error", {}) or {}).get("message") or j.get("message") or detail
    except (ValueError, AttributeError):
        pass
    detail = str(detail)[:300]
    hint = {
        400: "请求参数有误（模型名是否正确？max_tokens 是否超出该模型上限？）",
        401: "API Key 无效或已过期",
        403: "没有访问权限（Key 权限不足、区域限制或余额问题）",
        404: "接口地址或模型名不存在（检查 Base URL 与模型名）",
        429: "请求过于频繁或额度不足",
    }.get(status, "服务端错误，稍后重试" if status >= 500 else "请求失败")
    return f"[{status}] {hint}。服务返回：{detail}"


def _headers(cfg: LLMConfig) -> dict[str, str]:
    key = cfg.effective_key()
    if cfg.protocol == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    h = {"content-type": "application/json"}
    if key:
        h["authorization"] = f"Bearer {key}"
    return h


def _url(cfg: LLMConfig) -> str:
    base = cfg.base_url.rstrip("/")
    if cfg.protocol == "anthropic":
        return base + ("/messages" if base.endswith("/v1") else "/v1/messages")
    return base + "/chat/completions"


def _payload(cfg: LLMConfig, messages: list[dict], system: str | None, stream: bool = True) -> dict:
    if cfg.protocol == "anthropic":
        p = {"model": cfg.model, "max_tokens": cfg.max_tokens, "messages": messages, "stream": stream,
             "temperature": cfg.temperature}
        if system:
            p["system"] = system
        return p
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    return {"model": cfg.model, "messages": msgs, "stream": stream, "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens}


def _delta(cfg: LLMConfig, data: dict) -> str:
    if cfg.protocol == "anthropic":
        if data.get("type") == "content_block_delta":
            d = data.get("delta", {})
            if d.get("type") == "text_delta":
                return d.get("text", "")
        if data.get("type") == "error":
            raise LLMError(f"模型返回错误：{data.get('error', {}).get('message', data)}")
        return ""
    ch = data.get("choices") or []
    if not ch:
        return ""
    return (ch[0].get("delta") or {}).get("content") or ""  # 忽略 reasoning_content


def stream_chat(cfg: LLMConfig, messages: list[dict], system: str | None = None,
                cancel: Callable[[], bool] | None = None,
                transport: httpx.BaseTransport | None = None) -> Iterator[str]:
    """逐段产出文本。出错抛 LLMError；cancel() 为真时安静地停止。"""
    ok, why = cfg.is_ready()
    if not ok:
        raise LLMError(why)
    timeout = httpx.Timeout(cfg.timeout, connect=20.0)
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            with client.stream("POST", _url(cfg), headers=_headers(cfg), json=_payload(cfg, messages, system)) as r:
                if r.status_code != 200:
                    r.read()
                    raise LLMError(_friendly(r.status_code, r.text, cfg))
                for line in r.iter_lines():
                    if cancel and cancel():
                        return
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue
                    text = _delta(cfg, obj)
                    if text:
                        yield text
    except httpx.ConnectError as e:
        raise LLMError(f"无法连接到 {cfg.base_url}：请检查网络/代理/接口地址（{e}）") from e
    except httpx.TimeoutException as e:
        raise LLMError("请求超时：模型响应过慢或网络不稳定，可在设置中调大超时时间") from e
    except httpx.HTTPError as e:
        raise LLMError(f"网络错误：{type(e).__name__}: {e}") from e


def chat(cfg: LLMConfig, messages: list[dict], system: str | None = None, **kw) -> str:
    return "".join(stream_chat(cfg, messages, system, **kw))


def test_connection(cfg: LLMConfig, transport: httpx.BaseTransport | None = None) -> str:
    """发一个极短请求验证配置，返回模型回复（出错抛 LLMError）。"""
    small = LLMConfig(**{**cfg.__dict__, "max_tokens": 32, "timeout": 30.0})
    return chat(small, [{"role": "user", "content": "只回复两个字：成功"}], None, transport=transport).strip()
