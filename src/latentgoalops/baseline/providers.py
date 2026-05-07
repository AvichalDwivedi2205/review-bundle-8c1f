"""Provider adapters with retry and checkpoint-friendly errors."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


GRADIENT_BASE_URL = "https://inference.do-ai.run/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL_PRICING = {
    "openai-gpt-oss-20b": {"input": 0.05 / 1_000_000, "output": 0.45 / 1_000_000},
    "openai-gpt-oss-120b": {"input": 0.10 / 1_000_000, "output": 0.70 / 1_000_000},
    "kimi-k2.5": {"input": 0.50 / 1_000_000, "output": 2.70 / 1_000_000},
    "glm-5": {"input": 1.00 / 1_000_000, "output": 3.20 / 1_000_000},
    "openai/gpt-oss-20b": {"input": 0.03 / 1_000_000, "output": 0.11 / 1_000_000},
    "openai/gpt-oss-120b": {"input": 0.09 / 1_000_000, "output": 0.35 / 1_000_000},
    "alibaba-qwen3-32b": {"input": 0.25 / 1_000_000, "output": 0.55 / 1_000_000},
    "deepseek-r1-distill-llama-70b": {"input": 0.99 / 1_000_000, "output": 0.99 / 1_000_000},
}
GRADIENT_MODEL_PRICING = {
    "openai-gpt-oss-20b": {"input": 0.05 / 1_000_000, "output": 0.45 / 1_000_000},
    "openai-gpt-oss-120b": {"input": 0.10 / 1_000_000, "output": 0.70 / 1_000_000},
    "openai-gpt-5.4": {"input": 2.50 / 1_000_000, "output": 15.00 / 1_000_000},
    "openai-gpt-5.4-mini": {"input": 0.75 / 1_000_000, "output": 4.50 / 1_000_000},
    "openai-gpt-5.4-nano": {"input": 0.20 / 1_000_000, "output": 1.25 / 1_000_000},
    "openai-gpt-5.4-pro": {"input": 30.00 / 1_000_000, "output": 180.00 / 1_000_000},
    "anthropic-claude-4.6-sonnet": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "kimi-k2.5": {"input": 0.50 / 1_000_000, "output": 2.70 / 1_000_000},
    "glm-5": {"input": 1.00 / 1_000_000, "output": 3.20 / 1_000_000},
}
OPENROUTER_MODEL_PRICING = {
    "openai/gpt-oss-20b": {"input": 0.03 / 1_000_000, "output": 0.11 / 1_000_000},
    "openai/gpt-oss-120b": {"input": 0.09 / 1_000_000, "output": 0.35 / 1_000_000},
    "google/gemini-3-flash-preview": {"input": 0.50 / 1_000_000, "output": 3.00 / 1_000_000},
    "~google/gemini-flash-latest": {"input": 0.50 / 1_000_000, "output": 3.00 / 1_000_000},
    "google/gemini-3.1-flash-lite-preview": {"input": 0.25 / 1_000_000, "output": 1.50 / 1_000_000},
    "google/gemini-2.5-flash": {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
    "openai/gpt-5.4-mini": {"input": 0.75 / 1_000_000, "output": 4.50 / 1_000_000},
    "anthropic/claude-sonnet-4.6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
}
OPENROUTER_MODEL_ALIASES = {
    "openai-gpt-oss-20b": "openai/gpt-oss-20b",
    "openai-gpt-oss-120b": "openai/gpt-oss-120b",
}
GRADIENT_MODEL_ALIASES = {value: key for key, value in OPENROUTER_MODEL_ALIASES.items()}
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0


def _supports_reasoning_effort(request_model_name: str) -> bool:
    """Return whether the provider should request server-side reasoning controls."""
    return "gpt-oss" in request_model_name.lower()


def _json_response_format_enabled() -> bool:
    """Return whether the provider should request server-side JSON mode when possible."""
    value = os.getenv("MODEL_JSON_RESPONSE_FORMAT", "on").strip().lower()
    return value not in {"0", "false", "off", "disable", "disabled"}


def _uses_responses_api(base_url: str, request_model_name: str) -> bool:
    """Return whether Gradient requires the OpenAI Responses API."""
    return _is_gradient_endpoint(base_url) and request_model_name.startswith("openai-gpt-5")


class ProviderExhaustedError(RuntimeError):
    """Raised when billing or account limits stop the run."""


class ProviderCredentialError(RuntimeError):
    """Raised when the API token is invalid or revoked."""


class ProviderTransientError(RuntimeError):
    """Raised when a temporary provider failure should be resumed later."""


@dataclass(slots=True)
class ProviderResponse:
    """Structured provider response."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str | None = None
    raw: dict | None = None


class ChatProvider(Protocol):
    """Minimal provider protocol used by the benchmark runner."""

    model_name: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 700,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ProviderResponse: ...


def _is_gradient_endpoint(base_url: str) -> bool:
    return "inference.do-ai.run" in base_url


def _is_openrouter_endpoint(base_url: str) -> bool:
    return "openrouter.ai" in base_url


def _default_provider_base_url() -> str:
    return (
        os.getenv("API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or (OPENROUTER_BASE_URL if os.getenv("OPENROUTER_API_KEY") else GRADIENT_BASE_URL)
    )


def _default_ollama_host() -> str:
    return os.getenv("OLLAMA_HOST") or OLLAMA_BASE_URL


def _resolve_provider_token(base_url: str) -> tuple[str | None, str | None]:
    if _is_gradient_endpoint(base_url):
        candidates = [
            ("MODEL_ACCESS_KEY", os.getenv("MODEL_ACCESS_KEY")),
            ("DIGITALOCEAN_API_TOKEN", os.getenv("DIGITALOCEAN_API_TOKEN")),
            ("HF_TOKEN", os.getenv("HF_TOKEN")),
            ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
            ("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY")),
        ]
    elif _is_openrouter_endpoint(base_url):
        candidates = [
            ("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY")),
            ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
            ("MODEL_ACCESS_KEY", os.getenv("MODEL_ACCESS_KEY")),
            ("HF_TOKEN", os.getenv("HF_TOKEN")),
            ("DIGITALOCEAN_API_TOKEN", os.getenv("DIGITALOCEAN_API_TOKEN")),
        ]
    else:
        candidates = [
            ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
            ("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY")),
            ("MODEL_ACCESS_KEY", os.getenv("MODEL_ACCESS_KEY")),
            ("HF_TOKEN", os.getenv("HF_TOKEN")),
            ("DIGITALOCEAN_API_TOKEN", os.getenv("DIGITALOCEAN_API_TOKEN")),
        ]
    for source, value in candidates:
        if value:
            return value, source
    return None, None


def _resolve_provider_model_name(model_name: str, base_url: str) -> str:
    if _is_openrouter_endpoint(base_url):
        return OPENROUTER_MODEL_ALIASES.get(model_name, model_name)
    if _is_gradient_endpoint(base_url):
        return GRADIENT_MODEL_ALIASES.get(model_name, model_name)
    return model_name


def _resolve_model_pricing(model_name: str, request_model_name: str, base_url: str) -> dict[str, float]:
    candidate_keys = [request_model_name, model_name]
    pricing_table = DEFAULT_MODEL_PRICING
    if _is_openrouter_endpoint(base_url):
        pricing_table = {**DEFAULT_MODEL_PRICING, **OPENROUTER_MODEL_PRICING}
    elif _is_gradient_endpoint(base_url):
        pricing_table = {**DEFAULT_MODEL_PRICING, **GRADIENT_MODEL_PRICING}
    for key in candidate_keys:
        if key in pricing_table:
            return pricing_table[key]
    return {"input": 0.0, "output": 0.0}


class GradientChatProvider:
    """Chat-completions wrapper for OpenAI-compatible endpoints."""

    def __init__(self, model_name: str, max_retries: int = 6, temperature: float = 0.0, base_url: str | None = None) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.temperature = temperature
        self.request_timeout_seconds = float(
            os.getenv("MODEL_REQUEST_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS))
        )
        resolved_base_url = base_url or _default_provider_base_url()
        token, token_source = _resolve_provider_token(resolved_base_url)
        self.base_url = resolved_base_url
        self.token_source = token_source
        self.request_model_name = _resolve_provider_model_name(model_name, resolved_base_url)
        if not token:
            raise ValueError(
                "Set HF_TOKEN (or DIGITALOCEAN_API_TOKEN / OPENAI_API_KEY / MODEL_ACCESS_KEY / OPENROUTER_API_KEY) "
                "before running model baselines."
            )
        self.client = OpenAI(base_url=resolved_base_url, api_key=token, timeout=self.request_timeout_seconds, max_retries=0)

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    @classmethod
    def _responses_input(cls, messages: list[dict[str, Any]]) -> tuple[str | None, str]:
        instructions = "\n\n".join(
            cls._message_text(message)
            for message in messages
            if str(message.get("role", "")).lower() in {"system", "developer"}
        )
        prompt_parts = [
            f"{message.get('role', 'user')}: {cls._message_text(message)}"
            for message in messages
            if str(message.get("role", "")).lower() not in {"system", "developer"}
        ]
        return instructions or None, "\n\n".join(prompt_parts)

    def _complete_with_responses_api(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        reasoning_effort: str | None,
        response_format_enabled: bool,
    ) -> ProviderResponse:
        instructions, prompt_input = self._responses_input(messages)
        request_kwargs: dict[str, Any] = {
            "model": self.request_model_name,
            "input": prompt_input,
            "max_output_tokens": max_tokens,
            "timeout": self.request_timeout_seconds,
        }
        if instructions:
            request_kwargs["instructions"] = instructions
        if self.temperature != 0.0:
            request_kwargs["temperature"] = self.temperature
        if reasoning_effort is not None:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}
        if response_format_enabled:
            request_kwargs["text"] = {"format": {"type": "json_object"}}

        response = self.client.responses.create(**request_kwargs)
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        pricing = _resolve_model_pricing(self.model_name, self.request_model_name, self.base_url)
        cost_usd = input_tokens * pricing["input"] + output_tokens * pricing["output"]
        content = getattr(response, "output_text", None)
        if not content:
            content_parts: list[str] = []
            for output in getattr(response, "output", []) or []:
                for part in getattr(output, "content", []) or []:
                    part_type = getattr(part, "type", None)
                    if part_type not in {None, "output_text"}:
                        continue
                    text = getattr(part, "text", None)
                    if text:
                        content_parts.append(str(text))
            content = "".join(content_parts) or "{}"
        finish_reason = getattr(response, "status", None)
        return ProviderResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            finish_reason=finish_reason,
            raw=response.model_dump(mode="json"),
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 700,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        last_error: Exception | None = None
        response_format_enabled = _json_response_format_enabled()
        for attempt in range(self.max_retries):
            try:
                if _uses_responses_api(self.base_url, self.request_model_name):
                    return self._complete_with_responses_api(
                        messages, max_tokens, reasoning_effort, response_format_enabled
                    )
                request_kwargs: dict[str, Any] = {
                    "model": self.request_model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "timeout": self.request_timeout_seconds,
                }
                if reasoning_effort is not None and _supports_reasoning_effort(self.request_model_name):
                    request_kwargs["reasoning_effort"] = reasoning_effort
                if response_format_enabled:
                    request_kwargs["response_format"] = {"type": "json_object"}
                if _is_openrouter_endpoint(self.base_url):
                    request_kwargs["max_tokens"] = max_tokens
                else:
                    request_kwargs["max_completion_tokens"] = max_tokens
                response = self.client.chat.completions.create(**request_kwargs)
                usage = response.usage
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                pricing = _resolve_model_pricing(self.model_name, self.request_model_name, self.base_url)
                cost_usd = input_tokens * pricing["input"] + output_tokens * pricing["output"]
                choice = response.choices[0]
                message = choice.message
                content = message.content or "{}"
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return ProviderResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    finish_reason=getattr(choice, "finish_reason", None),
                    raw=response.model_dump(mode="json"),
                )
            except APIStatusError as exc:
                last_error = exc
                message = str(exc).lower()
                if exc.status_code == 400 and response_format_enabled and (
                    "response_format" in message
                    or "json_object" in message
                    or "json schema" in message
                    or "text.format" in message
                ):
                    response_format_enabled = False
                    continue
                if exc.status_code in (401, 403):
                    hint = ""
                    if (
                        _is_gradient_endpoint(self.base_url)
                        and self.token_source == "HF_TOKEN"
                        and os.getenv("DIGITALOCEAN_API_TOKEN")
                    ):
                        hint = (
                            " HF_TOKEN was selected before DIGITALOCEAN_API_TOKEN; "
                            "set HF_TOKEN equal to the DigitalOcean token or unset HF_TOKEN."
                        )
                    raise ProviderCredentialError(
                        f"Provider rejected {self.token_source or 'configured'} credentials for {self.base_url}.{hint}"
                    ) from exc
                if exc.status_code == 429 or 500 <= exc.status_code < 600:
                    if "quota" in message or "billing" in message or "credit" in message:
                        raise ProviderExhaustedError("Provider reported quota or billing exhaustion.") from exc
                    retry_after = 0.0
                    if getattr(exc, "response", None) is not None:
                        retry_value = exc.response.headers.get("retry-after")
                        if retry_value:
                            try:
                                retry_after = float(retry_value)
                            except ValueError:
                                retry_after = 0.0
                    if exc.status_code == 429 and retry_after <= 0.0:
                        retry_after = min(15.0 * (2**attempt), 120.0)
                    elif retry_after <= 0.0:
                        retry_after = min(2**attempt, 12)
                    time.sleep(retry_after)
                    continue
                raise
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 12))
                continue
        raise ProviderTransientError(f"Provider request failed after retries: {last_error}")


class OllamaChatProvider:
    """Native Ollama chat wrapper with schema-constrained JSON outputs."""

    def __init__(
        self,
        model_name: str,
        *,
        host: str | None = None,
        max_retries: int = 4,
        temperature: float = 0.0,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
        think: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.host = (host or _default_ollama_host()).rstrip("/")
        self.max_retries = max_retries
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.think = think
        self.request_timeout_seconds = float(
            os.getenv("MODEL_REQUEST_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS))
        )
        self.client = httpx.Client(base_url=self.host, timeout=self.request_timeout_seconds)

    def _resolve_think(self, reasoning_effort: str | None) -> bool | str | None:
        mode = (self.think or "auto").strip().lower()
        if mode in {"", "auto"}:
            return False
        if mode in {"on", "true"}:
            return True
        if mode in {"off", "false"}:
            return False
        if mode in {"low", "medium", "high"}:
            return mode
        if reasoning_effort is not None:
            return reasoning_effort
        return False

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 700,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = int(self.num_ctx)
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        think = self._resolve_think(reasoning_effort)
        if think is not None:
            payload["think"] = think
        if json_schema is not None:
            payload["format"] = json_schema
        elif _json_response_format_enabled():
            payload["format"] = "json"

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                message = data.get("message", {}) or {}
                content = message.get("content") or "{}"
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return ProviderResponse(
                    content=str(content),
                    input_tokens=int(data.get("prompt_eval_count", 0) or 0),
                    output_tokens=int(data.get("eval_count", 0) or 0),
                    cost_usd=0.0,
                    finish_reason=str(data.get("done_reason")) if data.get("done_reason") is not None else None,
                    raw=data,
                )
            except httpx.HTTPStatusError as exc:
                last_error = exc
                message = exc.response.text.lower()
                if exc.response.status_code in {404, 400} and "model" in message and "not found" in message:
                    raise ValueError(
                        f"Ollama model '{self.model_name}' is not available on {self.host}. Pull it first with `ollama pull {self.model_name}`."
                    ) from exc
                if exc.response.status_code in {408, 429} or 500 <= exc.response.status_code < 600:
                    time.sleep(min(2**attempt, 12))
                    continue
                raise
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 12))
                continue
        raise ProviderTransientError(f"Ollama request failed after retries: {last_error}")


def create_chat_provider(
    model_name: str,
    *,
    provider: str = "openai_compat",
    temperature: float = 0.0,
    base_url: str | None = None,
    ollama_host: str | None = None,
    ollama_num_ctx: int | None = None,
    ollama_keep_alive: str | None = None,
    ollama_think: str = "auto",
) -> ChatProvider:
    """Construct the configured provider backend."""
    resolved_provider = provider
    if resolved_provider == "auto":
        resolved_provider = "ollama" if (ollama_host or os.getenv("OLLAMA_HOST")) else "openai_compat"
    if resolved_provider == "ollama":
        return OllamaChatProvider(
            model_name,
            host=ollama_host,
            temperature=temperature,
            num_ctx=ollama_num_ctx,
            keep_alive=ollama_keep_alive,
            think=ollama_think,
        )
    return GradientChatProvider(model_name, temperature=temperature, base_url=base_url)
