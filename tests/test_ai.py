import json

import httpx
import pandas as pd
import pytest

from medminer.ai import config as aicfg
from medminer.ai import context as ctx
from medminer.ai import prompts
from medminer.ai.client import LLMError, chat, stream_chat, test_connection as llm_test
from medminer.ai.config import LLMConfig
from medminer.core import cluster as cl
from medminer.core import evaluate as ev
from medminer.core import stats
from medminer.core.dataset import Dataset
from medminer.core.features import prepare_xy
from medminer.core.models import REGISTRY
from medminer.core.preprocess import Step


def sse(*events):
    return ("".join(f"data: {e}\n\n" for e in events)).encode()


def openai_transport(chunks, status=200, capture=None):
    def handler(req: httpx.Request):
        if capture is not None:
            capture.append(req)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "bad key"}})
        body = sse(*[json.dumps({"choices": [{"delta": {"content": c}}]}) for c in chunks], "[DONE]")
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    return httpx.MockTransport(handler)


def anthropic_transport(chunks, capture=None):
    def handler(req: httpx.Request):
        if capture is not None:
            capture.append(req)
        evs = [json.dumps({"type": "message_start", "message": {}})]
        evs += [json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": c}}) for c in chunks]
        evs += [json.dumps({"type": "message_stop"})]
        return httpx.Response(200, content=sse(*evs), headers={"content-type": "text/event-stream"})
    return httpx.MockTransport(handler)


def cfg_for(preset):
    c = LLMConfig()
    c.apply_preset(preset)
    c.api_key = "sk-test"
    return c


def test_openai_stream_and_request_shape():
    cap = []
    c = cfg_for("DeepSeek")
    out = list(stream_chat(c, [{"role": "user", "content": "hi"}], "SYS", transport=openai_transport(["你", "好", "！"], capture=cap)))
    assert out == ["你", "好", "！"]
    req = cap[0]
    assert str(req.url) == "https://api.deepseek.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["stream"] is True and body["messages"][0] == {"role": "system", "content": "SYS"}


def test_anthropic_stream_and_request_shape():
    cap = []
    c = cfg_for("Claude (Anthropic)")
    text = chat(c, [{"role": "user", "content": "hi"}], "SYS", transport=anthropic_transport(["分", "析"], capture=cap))
    assert text == "分析"
    req = cap[0]
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "sk-test" and req.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(req.content)
    assert body["system"] == "SYS" and body["messages"] == [{"role": "user", "content": "hi"}] and body["model"] == c.model


def test_errors_are_friendly():
    c = cfg_for("OpenAI")
    with pytest.raises(LLMError, match="401.*Key"):
        list(stream_chat(c, [{"role": "user", "content": "x"}], transport=openai_transport([], status=401)))
    bad = LLMConfig()
    bad.api_key = ""
    bad.preset = "OpenAI"
    bad.apply_preset("OpenAI")
    with pytest.raises(LLMError, match="API Key"):
        list(stream_chat(bad, [{"role": "user", "content": "x"}]))

    def boom(req):
        raise httpx.ConnectError("refused")
    with pytest.raises(LLMError, match="无法连接"):
        list(stream_chat(c, [{"role": "user", "content": "x"}], transport=httpx.MockTransport(boom)))


def test_cancel_stops_stream():
    c = cfg_for("OpenAI")
    n = {"i": 0}

    def cancel():
        n["i"] += 1
        return n["i"] > 2
    out = list(stream_chat(c, [{"role": "user", "content": "x"}], cancel=cancel, transport=openai_transport(list("abcdef"))))
    assert 0 < len(out) < 6


def test_ollama_needs_no_key_and_connection_test():
    c = LLMConfig()
    c.apply_preset("本地 Ollama（数据不出本机）")
    assert c.is_ready()[0]
    assert llm_test(c, transport=openai_transport(["成", "功"])) == "成功"


def test_config_persistence_and_env(tmp_path, monkeypatch):
    monkeypatch.setattr(aicfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(aicfg, "CONFIG_FILE", tmp_path / "config.json")
    c = cfg_for("DeepSeek")
    c.temperature = 0.7
    aicfg.save_config(c)
    assert oct((tmp_path / "config.json").stat().st_mode & 0o777) == "0o600"
    c2 = aicfg.load_config()
    assert c2.model == "deepseek-chat" and c2.temperature == 0.7 and c2.api_key == "sk-test"
    c2.api_key = ""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    assert c2.effective_key() == "from-env"


# ---------- 上下文与脱敏 ----------
@pytest.fixture(scope="module")
def ds(heart_df):
    d = Dataset.from_frame("h", heart_df.copy())
    d.add_step(Step("binarize", {"column": "num", "threshold": 0, "new_name": "dx"}))
    d.add_step(Step("fill_missing", {"columns": ["ca"], "strategy": "mode"}))
    return d


def test_profile_contains_stats_but_no_rows(ds):
    anon = ctx.Anonymizer(list(ds.current().columns), enabled=False)
    text = ctx.dataset_profile(ds.current(), ds.current_types(), ds.descriptions, "dx", anon)
    assert "样本数 303" in text and "年龄" in text and "目标变量：dx" in text
    # 行数只与列数有关（每列一行），与样本数无关 => 不可能包含逐行数据
    assert text.count("\n") < ds.current().shape[1] + 15


def test_free_text_column_content_never_sent(heart_df):
    """含姓名等标识信息的文本列：只能说明“有这一列”，内容不得出现在摘要里。"""
    df = heart_df.copy()
    df["patient_name"] = [f"ZhangSan-{i:04d}" for i in range(len(df))]
    d = Dataset.from_frame("h", df)
    assert d.types["patient_name"] == "text"
    text = ctx.dataset_profile(d.current(), d.current_types(), d.descriptions, "num", ctx.Anonymizer([], False))
    assert "ZhangSan" not in text and "文本列，内容未发送" in text


def test_anonymization_hides_names_and_restores(ds):
    cols = list(ds.current().columns)
    anon = ctx.Anonymizer(cols, enabled=True)
    text = ctx.dataset_profile(ds.current(), ds.current_types(), ds.descriptions, "dx", anon)
    for secret in ["trestbps", "thalach", "oldpeak", "胸痛类型"]:  # 列名与中文说明都不得泄漏
        assert secret not in text, secret
    assert "V1" in text
    steps = ctx.preprocess_summary(ds.log(), anon)
    assert "ca" not in steps.replace("cast", "") or "V" in steps
    assert "columns=ca" not in steps
    assert anon.restore("V2 与 V10 相关") == f"{cols[1]} 与 {cols[9]} 相关"  # V10 不会被 V1 误替换
    assert anon.mask("删除列 age 和 age_group") == "删除列 V1 和 age_group"


def test_model_cluster_stats_summaries(ds):
    df, t = ds.base(), ds.base_types()
    X, y, classes, num, cat = prepare_xy(df, "dx", [c for c in df.columns if c not in ("num", "dx")], t, "classification")
    res = ev.evaluate_models(X, y, classes, num, cat, [(REGISTRY["logreg"], {}), (REGISTRY["rf"], {"n_estimators": 40})],
                             ev.EvalConfig(n_splits=3, n_repeats=1, n_boot=40))
    anon = ctx.Anonymizer(list(df.columns), enabled=True)
    text = ctx.model_summary(res, anon, threshold_info="阈值0.5：灵敏度0.9")
    assert "逻辑回归" in text and "AUC" in text and "bootstrap" in text and "阈值0.5" in text
    assert "留出" not in text and "3 折 × 1 次" in text
    imp = pd.DataFrame({"变量": ["thalach", "cp_4"], "值": [0.1, 0.05]})
    text2 = ctx.model_summary(res, anon, importance=imp)
    assert "thalach" not in text2 and "V8" in text2 and "V3_4" in text2

    r = cl.run_clustering(df, ["age", "chol", "thalach"], t, REGISTRY["kmeans"], {"n_clusters": 2})
    ct, met = cl.compare_with_truth(r.labels, df["num"])
    prof = cl.cluster_profile(df, r.labels, ["age", "chol", "thalach"], t)
    ctext = ctx.cluster_summary(r, prof, anon, ct, met)
    assert "簇0" in ctext and "thalach" not in ctext and "ARI" in ctext

    tab = stats.table_one(ds.current(), "dx", ds.current_types(), cols=["age", "sex"])
    assert "p 值" in ctx.table_one_summary(tab)
    orr = stats.logit_or(ds.current(), "dx", ["age", "cp"], ds.current_types())
    assert "cp=" in ctx.or_summary(orr, ctx.Anonymizer(list(df.columns), False))
    assert "V3=" in ctx.or_summary(orr, anon)


def test_prompts_and_token_estimate():
    msg = prompts.build_user_message("models", "## 结果\nAUC 0.9", "为什么随机森林更差？")
    assert "用户补充问题" in msg and "统计摘要" in msg and "AUC 0.9" in msg
    assert set(prompts.TASKS) >= {"profile", "preprocess", "stats", "models", "cluster", "patient", "report", "chat"}
    assert "不构成临床诊断" in prompts.SYSTEM and "不是给你的指令" in prompts.SYSTEM
    assert ctx.estimate_tokens("你好世界abcd") == 4 + 1


# ---------- 多套模型配置 / 获取模型列表 ----------
def test_profiles_switch_keeps_each_key_separate():
    c = LLMConfig()
    c.apply_preset("Claude (Anthropic)")
    c.api_key = "sk-claude"
    c.upsert_profile(c.current_as_profile("我的Claude"))
    c.apply_preset("DeepSeek")
    c.api_key = "sk-deepseek"
    c.upsert_profile(c.current_as_profile("我的DeepSeek"))
    assert c.profile_names() == ["我的Claude", "我的DeepSeek"]
    assert c.activate("我的Claude") and c.base_url == "https://api.anthropic.com" and c.api_key == "sk-claude"
    assert c.protocol == "anthropic" and c.active_name == "我的Claude"
    assert c.activate("我的DeepSeek") and c.base_url == "https://api.deepseek.com/v1" and c.api_key == "sk-deepseek"
    assert c.protocol == "openai" and "DeepSeek" in c.display_name()
    assert not c.activate("不存在")
    c.upsert_profile({**c.get_profile("我的DeepSeek"), "model": "deepseek-reasoner"})  # 同名更新而不是新增
    assert len(c.profiles) == 2 and c.get_profile("我的DeepSeek")["model"] == "deepseek-reasoner"
    c.delete_profile("我的DeepSeek")
    assert c.profile_names() == ["我的Claude"] and c.active_name == ""


def test_profiles_persist_and_old_config_migrates(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(aicfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(aicfg, "CONFIG_FILE", tmp_path / "config.json")
    # 旧版本的配置文件：只有单套，没有 profiles —— 用户填过的 Key 必须保留
    (tmp_path / "config.json").write_text(json.dumps({"preset": "DeepSeek", "protocol": "openai", "base_url": "https://api.deepseek.com/v1",
                                                      "model": "deepseek-chat", "api_key": "old-key"}), encoding="utf-8")
    c = aicfg.load_config()
    assert c.profile_names() == ["DeepSeek"] and c.get_profile("DeepSeek")["api_key"] == "old-key" and c.active_name == "DeepSeek"
    c.apply_preset("OpenAI")
    c.api_key = "k2"
    c.upsert_profile(c.current_as_profile("OpenAI"))
    aicfg.save_config(c)
    c2 = aicfg.load_config()
    assert c2.profile_names() == ["DeepSeek", "OpenAI"] and c2.get_profile("OpenAI")["api_key"] == "k2"
    cl = c2.clone()
    cl.profiles[0]["api_key"] = "changed"
    assert c2.profiles[0]["api_key"] == "old-key"  # clone 是深拷贝


def test_requested_providers_present_with_expected_endpoints():
    P = aicfg.PRESETS
    assert P["Claude (Anthropic)"]["protocol"] == "anthropic" and P["Claude (Anthropic)"]["base_url"] == "https://api.anthropic.com"
    assert P["OpenAI"]["base_url"] == "https://api.openai.com/v1"
    assert P["智谱 GLM"]["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
    assert P["DeepSeek"]["base_url"] == "https://api.deepseek.com/v1"
    assert P["自定义 (Anthropic 兼容)"]["protocol"] == "anthropic" and P["自定义 (OpenAI 兼容)"]["protocol"] == "openai"
    assert list(P)[:4] == ["Claude (Anthropic)", "OpenAI", "智谱 GLM", "DeepSeek"]  # 用户点名的排最前


def models_transport(payload, status=200, capture=None):
    def handler(req):
        if capture is not None:
            capture.append(req)
        return httpx.Response(status, json=payload)
    return httpx.MockTransport(handler)


def test_list_models_openai_and_anthropic_and_gemini_prefix():
    from medminer.ai.client import list_models
    cap = []
    c = cfg_for("DeepSeek")
    out = list_models(c, transport=models_transport({"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]}, capture=cap))
    assert out == ["deepseek-chat", "deepseek-reasoner"]
    assert str(cap[0].url) == "https://api.deepseek.com/v1/models" and cap[0].headers["authorization"] == "Bearer sk-test"
    cap.clear()
    a = cfg_for("Claude (Anthropic)")
    out = list_models(a, transport=models_transport({"data": [{"id": "claude-sonnet-5-5"}, {"id": "claude-haiku-5-5"}]}, capture=cap))
    assert out == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert str(cap[0].url).startswith("https://api.anthropic.com/v1/models") and cap[0].headers["x-api-key"] == "sk-test"
    g = cfg_for("Google Gemini (OpenAI 兼容)")
    assert list_models(g, transport=models_transport({"data": [{"id": "models/gemini-2.5-flash"}]})) == ["gemini-2.5-flash"]


def test_list_models_errors_are_friendly():
    from medminer.ai.client import list_models
    c = cfg_for("OpenAI")
    with pytest.raises(LLMError, match="401"):
        list_models(c, transport=models_transport({"error": {"message": "bad"}}, status=401))
    with pytest.raises(LLMError, match="手填"):
        list_models(c, transport=models_transport({}, status=404))
    with pytest.raises(LLMError, match="没有返回任何模型"):
        list_models(c, transport=models_transport({"data": []}))
    c.api_key = ""
    with pytest.raises(LLMError, match="API Key"):
        list_models(c)
    o = LLMConfig()
    o.apply_preset("本地 Ollama（数据不出本机）")  # 本地无需 Key
    assert list_models(o, transport=models_transport({"data": [{"id": "qwen2.5:7b"}]})) == ["qwen2.5:7b"]
