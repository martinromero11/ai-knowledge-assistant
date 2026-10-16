"""Thin clients for LLM providers. Both expose the same complete() method."""

import logging
from typing import Protocol

import httpx

from .config import settings

log = logging.getLogger("app.llm")


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict]) -> str: ...


class AnthropicClient:
    def __init__(self, api_key: str, model: str | None = None, timeout: float = 30):
        self.api_key = api_key
        self.model = model or "claude-haiku-5-5"
        self.timeout = timeout

    def complete(self, system: str, messages: list[dict]) -> str:
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={"model": self.model, "max_tokens": 1024, "system": system, "messages": messages},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return "".join(b.get("text", "") for b in resp.json()["content"])
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Anthropic API returned {e.response.status_code}: {e.response.text[:200]}") from e
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LLMError(f"Anthropic API request failed: {type(e).__name__}: {e}") from e


class OpenAIClient:
    def __init__(self, api_key: str, model: str | None = None, timeout: float = 30):
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"
        self.timeout = timeout

    def complete(self, system: str, messages: list[dict]) -> str:
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0.2,
                    "messages": [{"role": "system", "content": system}, *messages],
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise LLMError(f"OpenAI API returned {e.response.status_code}: {e.response.text[:200]}") from e
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LLMError(f"OpenAI API request failed: {type(e).__name__}: {e}") from e


def default_client() -> LLMClient | None:
    provider = settings.llm_provider
    timeout = settings.llm_timeout_seconds
    if provider in ("", "anthropic") and settings.anthropic_api_key:
        return AnthropicClient(settings.anthropic_api_key, settings.llm_model, timeout)
    if provider in ("", "openai") and settings.openai_api_key:
        return OpenAIClient(settings.openai_api_key, settings.llm_model, timeout)
    return None
