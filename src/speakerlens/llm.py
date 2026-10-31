"""A small LLM client for translation and summaries.

DeepSeek and Ollama both speak the OpenAI chat-completions API, so one client covers
both. Configure it in .env (see .env.example):

    DEEPSEEK_API_KEY=sk-...        # cloud: transcript text is sent to DeepSeek
    LLM_PROVIDER=ollama            # or fully local, nothing leaves your machine
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Protocol

from speakerlens.config import LLMSettings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMTruncated(LLMError):
    """The reply hit the output-token limit, so the JSON is incomplete."""


class LLMUnavailable(LLMError):
    """No request can succeed until the setup changes: a rejected key, an empty account
    balance or an unknown model. Callers stop instead of splitting the work and retrying."""


class JSONChat(Protocol):
    def chat_json(self, system: str, user: str, temperature: float = 0.2) -> dict[str, Any]: ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
# Reasoning models served locally (e.g. Qwen3 on Ollama) may put their thinking inline.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_json_content(content: str) -> dict[str, Any]:
    """Parse a model reply as JSON, tolerating <think> blocks, ```json fences and text
    around the object."""
    text = _FENCE.sub("", _THINK.sub("", content).strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise LLMError("Expected a JSON object from the model.")
    return data


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider: str = "openai-compatible",
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 300.0,
        retries: int = 2,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self.retries = retries
        # The SDK already retries connection errors and rate limits with backoff.
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=3)

    @classmethod
    def from_settings(cls, llm: LLMSettings) -> LLMClient | None:
        if not llm.enabled:
            return None
        assert llm.base_url and llm.api_key and llm.model
        return cls(
            base_url=llm.base_url,
            api_key=llm.api_key,
            model=llm.model,
            provider=llm.provider,
            max_tokens=llm.max_tokens,
            extra_body=llm.extra_body,
        )

    def _request(self, system: str, user: str, temperature: float) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            **kwargs,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMTruncated("The reply was cut off at the output-token limit.")
        content = choice.message.content or ""
        if not content.strip():
            raise LLMError("The model returned an empty reply.")
        return parse_json_content(content)

    def chat_json(self, system: str, user: str, temperature: float = 0.2) -> dict[str, Any]:
        import openai

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._request(system, user, temperature)
            except openai.AuthenticationError as exc:
                raise LLMUnavailable(
                    f"{self.provider} rejected the API key. Check DEEPSEEK_API_KEY in .env."
                ) from exc
            except openai.NotFoundError as exc:
                raise LLMUnavailable(f"{self.provider} has no model {self.model!r}: {exc}") from exc
            except openai.BadRequestError as exc:
                raise LLMError(
                    f"{self.provider} refused the request for model {self.model!r}: {exc}"
                ) from exc
            except openai.APIStatusError as exc:
                if exc.status_code in (402, 403):
                    raise LLMUnavailable(
                        f"{self.provider} refused the request (HTTP {exc.status_code}): "
                        "check the account balance and permissions."
                    ) from exc
                last_error = exc
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(min(2**attempt, 10))
            except LLMTruncated:
                raise  # the caller can split the work into smaller pieces
            except Exception as exc:  # bad JSON, empty replies, network trouble
                last_error = exc
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(min(2**attempt, 10))
        raise LLMError(f"LLM call failed after {self.retries + 1} attempts: {last_error}")
