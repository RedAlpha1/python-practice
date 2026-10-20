import math
from functools import cache
from typing import Literal

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from ai_lab.config import get_settings
from ai_lab.models import Message

FinishReason = Literal["stop", "max_tokens", "safety", "other"]

_FINISH_MAP: dict[types.FinishReason, FinishReason] = {
    types.FinishReason.STOP: "stop",
    types.FinishReason.MAX_TOKENS: "max_tokens",
    types.FinishReason.SAFETY: "safety",
}

# Gemini calls the assistant "model"; our public vocabulary says "assistant".
_ROLE_MAP = {"user": "user", "assistant": "model"}


class LLMError(Exception):
    """Provider-independent API failure; status and retry_after may be unknown."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class LLMResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    finish_reason: FinishReason


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


def to_message(msg: ChatMessage) -> Message:
    return Message(role=msg.role, content=msg.content)


def from_message(msg: Message) -> ChatMessage:
    if msg.role == "system":
        raise ValueError("system messages are not part of a chat history")
    return ChatMessage(role=msg.role, content=msg.content)


@cache
def _client() -> genai.Client:
    return genai.Client(api_key=get_settings().gemini_api_key.get_secret_value())


def _normalize_finish(reason: types.FinishReason | None) -> FinishReason:
    return _FINISH_MAP.get(reason, "other") if reason else "other"


def _retry_after(details: object) -> float | None:
    """Seconds from Gemini's RetryInfo, e.g. {"error": {"details": [{"retryDelay": "52s"}]}}.

    Runs inside an except block, so it must never raise: anything unexpected is None.
    """
    if not isinstance(details, dict):
        return None
    error = details.get("error")
    if not isinstance(error, dict):
        return None
    items = error.get("details")
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        delay = item.get("retryDelay")
        if not isinstance(delay, str) or not delay.endswith("s"):
            continue
        try:
            seconds = float(delay.removesuffix("s"))
        except ValueError:
            return None
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    return None


def _generate(
    contents: str | list[types.Content],
    system: str | None,
    model: str | None,
    max_tokens: int | None,
    thinking_budget: int | None,
) -> LLMResponse:
    settings = get_settings()
    config = types.GenerateContentConfig(
        max_output_tokens=settings.max_tokens if max_tokens is None else max_tokens,
        system_instruction=system,
        thinking_config=(
            None
            if thinking_budget is None
            else types.ThinkingConfig(thinking_budget=thinking_budget)
        ),
    )
    try:
        response = _client().models.generate_content(
            model=settings.default_model if model is None else model,
            contents=contents,
            config=config,
        )
    except genai_errors.APIError as e:
        raise LLMError(
            e.message or str(e), status=e.code, retry_after=_retry_after(e.details)
        ) from e
    except httpx.TransportError as e:  # no HTTP response: DNS, connection, timeout
        raise LLMError(f"network error: {e!r}") from e

    usage = response.usage_metadata
    candidate = response.candidates[0] if response.candidates else None
    return LLMResponse(
        text=response.text or "",
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        thinking_tokens=(usage.thoughts_token_count or 0) if usage else 0,
        finish_reason=_normalize_finish(candidate.finish_reason if candidate else None),
    )


def ask(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
) -> LLMResponse:
    return _generate(prompt, system, model, max_tokens, thinking_budget)


def chat(
    history: list[ChatMessage],
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
) -> LLMResponse:
    """Send the whole conversation so far; the last message should be the user's."""
    contents = [
        types.Content(role=_ROLE_MAP[m.role], parts=[types.Part(text=m.content)])
        for m in history
    ]
    return _generate(contents, system, model, max_tokens, thinking_budget)
