"""Provider token-resolution tests."""

from __future__ import annotations

import latentgoalops.baseline.providers as providers
from latentgoalops.baseline.providers import (
    OPENROUTER_BASE_URL,
    GradientChatProvider,
    OllamaChatProvider,
    ProviderTransientError,
    create_chat_provider,
    _resolve_model_pricing,
    _default_provider_base_url,
    _resolve_provider_model_name,
    _resolve_provider_token,
)


def test_gradient_endpoint_prefers_digitalocean_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    token, source = _resolve_provider_token("https://inference.do-ai.run/v1")
    assert token == "do_token"
    assert source == "DIGITALOCEAN_API_TOKEN"


def test_gradient_endpoint_prefers_model_access_key(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    monkeypatch.setenv("MODEL_ACCESS_KEY", "model_access_key")
    token, source = _resolve_provider_token("https://inference.do-ai.run/v1")
    assert token == "model_access_key"
    assert source == "MODEL_ACCESS_KEY"


def test_non_gradient_endpoint_prefers_openai_key(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    monkeypatch.setenv("OPENAI_API_KEY", "openai_token")
    token, source = _resolve_provider_token("https://api.openai.com/v1")
    assert token == "openai_token"
    assert source == "OPENAI_API_KEY"


def test_openrouter_endpoint_prefers_openrouter_key(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_token")
    token, source = _resolve_provider_token(OPENROUTER_BASE_URL)
    assert token == "or_token"
    assert source == "OPENROUTER_API_KEY"


def test_openrouter_key_sets_default_base_url(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_token")
    assert _default_provider_base_url() == OPENROUTER_BASE_URL


def test_model_aliases_follow_provider_endpoint():
    assert _resolve_provider_model_name("openai-gpt-oss-20b", OPENROUTER_BASE_URL) == "openai/gpt-oss-20b"
    assert _resolve_provider_model_name("openai/gpt-oss-20b", "https://inference.do-ai.run/v1") == "openai-gpt-oss-20b"


def test_model_pricing_tracks_provider_endpoint():
    openrouter_pricing = _resolve_model_pricing("openai-gpt-oss-20b", "openai/gpt-oss-20b", OPENROUTER_BASE_URL)
    gradient_pricing = _resolve_model_pricing("openai-gpt-oss-20b", "openai-gpt-oss-20b", "https://inference.do-ai.run/v1")
    assert openrouter_pricing["input"] < gradient_pricing["input"]
    assert openrouter_pricing["output"] < gradient_pricing["output"]


def test_gradient_pricing_includes_new_do_models():
    kimi_pricing = _resolve_model_pricing("kimi-k2.5", "kimi-k2.5", "https://inference.do-ai.run/v1")
    glm_pricing = _resolve_model_pricing("glm-5", "glm-5", "https://inference.do-ai.run/v1")
    gpt54_pricing = _resolve_model_pricing("openai-gpt-5.4", "openai-gpt-5.4", "https://inference.do-ai.run/v1")
    sonnet_pricing = _resolve_model_pricing(
        "anthropic-claude-4.6-sonnet", "anthropic-claude-4.6-sonnet", "https://inference.do-ai.run/v1"
    )
    assert kimi_pricing["input"] > 0.0
    assert kimi_pricing["output"] > kimi_pricing["input"]
    assert glm_pricing["input"] > kimi_pricing["input"]
    assert glm_pricing["output"] > glm_pricing["input"]
    assert gpt54_pricing["output"] > gpt54_pricing["input"]
    assert sonnet_pricing["output"] > sonnet_pricing["input"]


def test_provider_client_uses_bounded_timeout(monkeypatch):
    captured: dict[str, float | int | str] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    monkeypatch.setenv("API_BASE_URL", "https://inference.do-ai.run/v1")
    monkeypatch.setenv("MODEL_REQUEST_TIMEOUT_SECONDS", "37")
    monkeypatch.setattr(providers, "OpenAI", FakeOpenAI)

    provider = providers.GradientChatProvider("openai-gpt-oss-20b")

    assert provider.request_timeout_seconds == 37.0
    assert captured["timeout"] == 37.0
    assert captured["max_retries"] == 0


def test_provider_requests_json_response_format(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Usage:
                prompt_tokens = 10
                completion_tokens = 5

            class Message:
                content = '{"task_id":"task1_feedback_triage","labels":[],"priorities":[],"escalate_ids":[]}'

            class Choice:
                message = Message()
                finish_reason = "stop"

            class Response:
                usage = Usage()
                choices = [Choice()]

                def model_dump(self, mode="json"):
                    return {"ok": True}

            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    monkeypatch.setenv("API_BASE_URL", "https://inference.do-ai.run/v1")
    monkeypatch.setattr(providers, "OpenAI", FakeOpenAI)

    provider = providers.GradientChatProvider("kimi-k2.5")
    response = provider.complete([{"role": "user", "content": "Return JSON"}], max_tokens=64)

    assert captured["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in captured
    assert response.finish_reason == "stop"


def test_provider_complete_accepts_explicit_reasoning_effort(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Usage:
                prompt_tokens = 10
                completion_tokens = 5

            class Message:
                content = '{"task_id":"task1_feedback_triage","labels":[],"priorities":[],"escalate_ids":[]}'

            class Choice:
                message = Message()
                finish_reason = "length"

            class Response:
                usage = Usage()
                choices = [Choice()]

                def model_dump(self, mode="json"):
                    return {"ok": True}

            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    monkeypatch.setenv("API_BASE_URL", "https://inference.do-ai.run/v1")
    monkeypatch.setattr(providers, "OpenAI", FakeOpenAI)

    provider = GradientChatProvider("openai-gpt-oss-20b")
    response = provider.complete(
        [{"role": "user", "content": "Return JSON"}],
        max_tokens=64,
        reasoning_effort="low",
    )

    assert captured["reasoning_effort"] == "low"
    assert response.finish_reason == "length"


def test_gradient_gpt5_models_use_responses_api(monkeypatch):
    captured: dict[str, object] = {}

    class Usage:
        input_tokens = 11
        output_tokens = 7

    class Response:
        usage = Usage()
        output_text = '{"ok":true}'
        status = "completed"

        def model_dump(self, mode="json"):
            return {"ok": True}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return Response()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()

    monkeypatch.setenv("MODEL_ACCESS_KEY", "model_access_key")
    monkeypatch.setenv("API_BASE_URL", "https://inference.do-ai.run/v1")
    monkeypatch.setattr(providers, "OpenAI", FakeOpenAI)

    provider = GradientChatProvider("openai-gpt-5.4-mini")
    response = provider.complete([{"role": "user", "content": "Return JSON"}], max_tokens=64)

    assert captured["model"] == "openai-gpt-5.4-mini"
    assert captured["input"] == "user: Return JSON"
    assert captured["max_output_tokens"] == 64
    assert captured["text"] == {"format": {"type": "json_object"}}
    assert response.content == '{"ok":true}'
    assert response.input_tokens == 11
    assert response.output_tokens == 7


def test_provider_complete_raises_transient_error_after_connection_retries(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            raise providers.APIConnectionError(message="temporary network issue", request=None)

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    monkeypatch.setenv("API_BASE_URL", "https://inference.do-ai.run/v1")
    monkeypatch.setattr(providers, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(providers.time, "sleep", lambda *_args, **_kwargs: None)

    provider = GradientChatProvider("openai-gpt-oss-20b", max_retries=2)
    try:
        provider.complete([{"role": "user", "content": "Return JSON"}], max_tokens=64)
    except ProviderTransientError as exc:
        assert "Provider request failed after retries" in str(exc)
    else:
        raise AssertionError("Expected transient provider failure.")


def test_create_chat_provider_selects_ollama():
    provider = create_chat_provider("gpt-oss:20b", provider="ollama", ollama_host="http://127.0.0.1:11434")
    assert isinstance(provider, OllamaChatProvider)
    assert provider.host == "http://127.0.0.1:11434"


def test_ollama_provider_posts_schema_payload(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": '{"selected":["item_1"]}'},
                "prompt_eval_count": 21,
                "eval_count": 9,
                "done_reason": "stop",
            }

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        def post(self, path, json):
            captured["path"] = path
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(providers.httpx, "Client", FakeClient)

    provider = OllamaChatProvider(
        "gpt-oss:20b",
        host="http://127.0.0.1:11434",
        temperature=0.1,
        num_ctx=32768,
        keep_alive="30m",
        think="off",
    )
    response = provider.complete(
        [{"role": "user", "content": "Return JSON"}],
        max_tokens=128,
        json_schema={"type": "object", "properties": {"selected": {"type": "array"}}},
    )

    assert captured["path"] == "/api/chat"
    assert captured["json"]["model"] == "gpt-oss:20b"
    assert captured["json"]["format"]["type"] == "object"
    assert captured["json"]["options"]["num_predict"] == 128
    assert captured["json"]["options"]["num_ctx"] == 32768
    assert captured["json"]["keep_alive"] == "30m"
    assert captured["json"]["think"] is False
    assert response.input_tokens == 21
    assert response.output_tokens == 9
    assert response.finish_reason == "stop"
