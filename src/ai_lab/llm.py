import math
from collections.abc import Iterator
from contextlib import contextmanager
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


class TextDelta(BaseModel):
    text: str


class StreamDone(BaseModel):
    response: LLMResponse


StreamEvent = TextDelta | StreamDone


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
            continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


def _build_config(
    system: str | None,
    max_tokens: int | None,
    thinking_budget: int | None,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        max_output_tokens=(
            get_settings().max_tokens if max_tokens is None else max_tokens
        ),
        system_instruction=system,
        thinking_config=(
            None
            if thinking_budget is None
            else types.ThinkingConfig(thinking_budget=thinking_budget)
        ),
    )


def _to_contents(history: list[ChatMessage]) -> list[types.Content]:
    return [
        types.Content(role=_ROLE_MAP[m.role], parts=[types.Part(text=m.content)])
        for m in history
    ]


@contextmanager
def _translate_errors() -> Iterator[None]:
    """Turn provider and network exceptions into LLMError."""
    try:
        yield
    except genai_errors.APIError as e:
        raise LLMError(
            e.message or str(e), status=e.code, retry_after=_retry_after(e.details)
        ) from e
    except httpx.TransportError as e:  # no HTTP response: DNS, connection, timeout
        raise LLMError(f"network error: {e!r}") from e


def _make_response(
    text: str,
    usage: types.GenerateContentResponseUsageMetadata | None,
    finish: types.FinishReason | None,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        thinking_tokens=(usage.thoughts_token_count or 0) if usage else 0,
        finish_reason=_normalize_finish(finish),
    )


def _model_name(model: str | None) -> str:
    return get_settings().default_model if model is None else model


def _generate(
    contents: str | list[types.Content],
    system: str | None,
    model: str | None,
    max_tokens: int | None,
    thinking_budget: int | None,
) -> LLMResponse:
    config = _build_config(system, max_tokens, thinking_budget)
    with _translate_errors():
        response = _client().models.generate_content(
            model=_model_name(model), contents=contents, config=config
        )

    candidate = response.candidates[0] if response.candidates else None
    return _make_response(
        response.text or "",
        response.usage_metadata,
        candidate.finish_reason if candidate else None,
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
    return _generate(_to_contents(history), system, model, max_tokens, thinking_budget)


def stream(
    history: list[ChatMessage],
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
) -> Iterator[StreamEvent]:
    """Yield a TextDelta per chunk of text, then one StreamDone with the full reply."""
    config = _build_config(system, max_tokens, thinking_budget)
    pieces: list[str] = []
    usage: types.GenerateContentResponseUsageMetadata | None = None
    finish: types.FinishReason | None = None

    # The error can strike at chunk 40, so the whole loop is inside the context.
    with _translate_errors():
        for chunk in _client().models.generate_content_stream(
            model=_model_name(model), contents=_to_contents(history), config=config
        ):
            if chunk.text:  # thinking chunks and the final chunk can have no text
                pieces.append(chunk.text)
                yield TextDelta(text=chunk.text)
            if chunk.usage_metadata:
                usage = chunk.usage_metadata
            if chunk.candidates and chunk.candidates[0].finish_reason:
                finish = chunk.candidates[0].finish_reason

    yield StreamDone(response=_make_response("".join(pieces), usage, finish))
