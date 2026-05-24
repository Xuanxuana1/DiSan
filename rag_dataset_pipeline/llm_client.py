#!/usr/bin/env python3
"""Small OpenAI-compatible chat-completions client used by the pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for an OpenAI-compatible chat endpoint."""

    base_url: str
    api_key: str
    model: str
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    temperature: float = 0.2

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.2,
    ) -> "LLMConfig":
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_model = model or os.environ.get("OPENAI_MODEL")

        missing = []
        if not resolved_base_url:
            missing.append("OPENAI_BASE_URL or --base-url")
        if not resolved_api_key:
            missing.append("OPENAI_API_KEY or --api-key")
        if not resolved_model:
            missing.append("OPENAI_MODEL or --model")
        if missing:
            raise ValueError(f"Missing LLM configuration: {', '.join(missing)}")

        return cls(
            base_url=resolved_base_url.rstrip("/"),
            api_key=resolved_api_key,
            model=resolved_model,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            temperature=temperature,
        )


def chat_completion(
    config: LLMConfig,
    *,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, int | None, int | None]:
    """Call a chat-completions endpoint and return text plus token usage."""

    payload: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(config.max_retries):
        try:
            response = requests.post(
                f"{config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage") or {}
            return (
                content,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 401:
                raise RuntimeError("LLM authentication failed; check API key") from exc
            last_error = exc
            if status_code is not None and status_code < 500 and status_code != 429:
                raise
        except requests.exceptions.RequestException as exc:
            last_error = exc

        if attempt < config.max_retries - 1:
            time.sleep(config.retry_delay)

    raise RuntimeError(f"LLM request failed after {config.max_retries} attempts") from last_error

