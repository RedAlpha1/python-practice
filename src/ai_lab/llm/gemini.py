"""Gemini backend: turns provider details into the types in ai_lab.llm.schemas."""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import assert_never

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from ai_lab.config import get_settings
from ai_lab.llm.retry import retry_llm
from ai_lab.llm.schemas import (
    AssistantTurn,
    ChatMessage,
    FinishReason,
    LLMError,
    LLMResponse,
    StreamDone,
    StreamEvent,
    StructuredOutputError,
    TextDelta,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    ToolSpec,
    Turn,
    UserText,
)

_FINISH_MAP: dict[types.FinishReason, FinishReason] = {
    types.FinishReason.STOP: "stop",
    types.FinishReason.MAX_TOKENS: "max_tokens",
    types.FinishReason.SAFETY: "safety",
}

# Gemini calls the assistant "model"; our public vocabulary says "assistant".
_ROLE_MAP = {"user": "user", "assistant": "model"}


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


@retry_llm()
def ask(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
) -> LLMResponse:
    return _generate(prompt, system, model, max_tokens, thinking_budget)


@retry_llm()
def chat(
    history: list[ChatMessage],
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
) -> LLMResponse:
    """Send the whole conversation so far; the last message should be the user's."""
    return _generate(_to_contents(history), system, model, max_tokens, thinking_budget)


# No @retry_llm here: by the time a stream fails, chunks are already on the user's
# screen, and a retry would replay the reply from the start.
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


def _tool_result_part(result: ToolResult) -> types.Part:
    # Gemini's convention: "error" key on failure, "output" key on success.
    response = (
        {"error": result.error}
        if result.error is not None
        else {"output": result.output}
    )
    return types.Part(
        function_response=types.FunctionResponse(
            id=result.call_id, name=result.name, response=response
        )
    )


def _turns_to_contents(turns: list[Turn]) -> list[types.Content]:
    contents: list[types.Content] = []
    for turn in turns:
        match turn:
            case UserText(text=text):
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=text)])
                )
            case AssistantTurn(provider_raw=types.Content() as raw):
                # Gemini's own turn, exactly as it came (keeps call ids and
                # thought signatures, which a rebuilt turn would lose).
                if not raw.parts:  # e.g. a safety block: Gemini would answer 400
                    raise ValueError("AssistantTurn has neither text nor tool calls")
                contents.append(raw)
            case AssistantTurn(text=text, tool_calls=calls):
                parts = [types.Part(text=text)] if text else []
                parts += [
                    types.Part(
                        function_call=types.FunctionCall(
                            id=c.id, name=c.name, args=c.args
                        )
                    )
                    for c in calls
                ]
                if not parts:
                    raise ValueError("AssistantTurn has neither text nor tool calls")
                contents.append(types.Content(role="model", parts=parts))
            case ToolResultsTurn(results=results):
                contents.append(
                    types.Content(
                        role="user", parts=[_tool_result_part(r) for r in results]
                    )
                )
            case _:
                assert_never(turn)  # a new Turn type must get its own case above
    return contents


def _tool_declaration(spec: ToolSpec) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=spec.name,
        description=spec.description,
        parameters_json_schema=spec.parameter,
    )


def _reply_text(content: types.Content | None) -> str:
    """Join the visible text parts. resp.text would warn when a function_call is there."""
    if content is None or not content.parts:
        return ""
    return "".join(p.text for p in content.parts if p.text and not p.thought)


@retry_llm()
def generate_turn(
    turns: list[Turn],
    tools: list[ToolSpec],
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
) -> tuple[AssistantTurn, LLMResponse]:
    """One model round: send the turns, get back the assistant's turn and usage.

    The model may answer with text, tool calls, or both. We do NOT run the tools:
    the caller does that and appends a ToolResultsTurn before the next round.
    """
    config = _build_config(system, max_tokens, thinking_budget)
    if tools:
        config.tools = [
            types.Tool(function_declarations=[_tool_declaration(t) for t in tools])
        ]
    # The SDK would otherwise run Python-callable tools itself; we run them ourselves.
    config.automatic_function_calling = types.AutomaticFunctionCallingConfig(
        disable=True
    )

    with _translate_errors():
        resp = _client().models.generate_content(
            model=_model_name(model),
            contents=_turns_to_contents(turns),
            config=config,
        )

    candidate = resp.candidates[0] if resp.candidates else None
    content = candidate.content if candidate else None
    text = _reply_text(content)
    tool_calls = [
        ToolCall(id=c.id, name=c.name or "", args=c.args or {})
        for c in resp.function_calls or []
    ]
    turn = AssistantTurn(text=text, tool_calls=tool_calls, provider_raw=content)
    response = _make_response(
        text, resp.usage_metadata, candidate.finish_reason if candidate else None
    )
    return turn, response


@retry_llm()
def ask_structured[T: BaseModel](
    prompt: str,
    schema: type[T],
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
) -> tuple[T, LLMResponse]:
    """Ask for JSON that matches `schema`; return the validated object and the usage."""
    config = _build_config(system, max_tokens, thinking_budget)
    config.response_mime_type = "application/json"
    config.response_json_schema = schema.model_json_schema()

    with _translate_errors():
        resp = _client().models.generate_content(
            model=_model_name(model), contents=prompt, config=config
        )

    candidate = resp.candidates[0] if resp.candidates else None
    text = _reply_text(candidate.content if candidate else None)
    response = _make_response(
        text, resp.usage_metadata, candidate.finish_reason if candidate else None
    )

    if response.finish_reason != "stop":  # cut off, blocked, or otherwise unfinished
        raise StructuredOutputError(
            f"reply is not a finished answer (finish_reason={response.finish_reason})",
            raw=text,
            response=response,
        )
    try:
        return schema.model_validate_json(text), response
    except ValidationError as e:
        raise StructuredOutputError(
            f"model output does not match {schema.__name__}: {e}",
            raw=text,
            response=response,
        ) from e
