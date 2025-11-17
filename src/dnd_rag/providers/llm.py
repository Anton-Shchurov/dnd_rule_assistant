"""LLM provider for OpenAI chat completions.

RU: Провайдер LLM для Chat Completions OpenAI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

_dotenv_path = find_dotenv(filename=".env", usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path, override=False)

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Single chat message compatible with OpenAI format."""

    role: str
    content: str


@dataclass
class LLMResponse:
    """Structured result returned by LLMClient."""

    content: str
    model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    raw: Any


class LLMClient:
    """Thin wrapper around OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = client or OpenAI()

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a chat completion for the supplied message history.

        RU: Генерирует ответ LLM для списка сообщений.
        """
        if not messages:
            raise ValueError("messages должны содержать хотя бы одно сообщение.")

        payload = [{"role": m.role, "content": m.content} for m in messages]
        eff_temperature = temperature if temperature is not None else self.temperature
        eff_max_tokens = max_tokens if max_tokens is not None else self.max_output_tokens

        logger.debug(
            "LLM request: model=%s, messages=%d, temperature=%s, max_tokens=%s",
            self.model,
            len(payload),
            eff_temperature,
            eff_max_tokens,
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=eff_temperature,
            max_tokens=eff_max_tokens,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)

        return LLMResponse(
            content=content.strip(),
            model=response.model or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
            raw=response,
        )


def build_message(role: str, content: str) -> ChatMessage:
    """Helper to create ChatMessage instances."""

    return ChatMessage(role=role, content=content)
