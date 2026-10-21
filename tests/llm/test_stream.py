import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from ai_lab.llm import ChatMessage, LLMError, LLMResponse, StreamDone, TextDelta, stream
from tests.llm.fakes import StreamingModels, chunk, error_body

HISTORY = [ChatMessage(role="user", content="hi")]


def test_yields_deltas_then_done(install_models):
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=5, candidates_token_count=7, thoughts_token_count=2
    )
    install_models(
        StreamingModels(
            [
                chunk("Hel"),
                chunk(None),  # a thinking chunk has no text: skipped
                chunk("lo"),
                chunk("", usage=usage, finish=types.FinishReason.STOP),
            ]
        )
    )

    events = list(stream(HISTORY))

    assert events[:-1] == [TextDelta(text="Hel"), TextDelta(text="lo")]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.response == LLMResponse(
        text="Hello",
        input_tokens=5,
        output_tokens=7,
        thinking_tokens=2,
        finish_reason="stop",
    )


def test_server_error_mid_stream_after_some_text(install_models):
    error = genai_errors.ServerError(503, error_body(503, "UNAVAILABLE"))
    install_models(StreamingModels([chunk("par"), chunk("tial")], error))

    seen = []
    with pytest.raises(LLMError) as info:
        seen.extend(stream(HISTORY))  # keeps what arrived before the error

    assert seen == [TextDelta(text="par"), TextDelta(text="tial")]
    assert info.value.status == 503


def test_timeout_mid_stream(install_models):
    install_models(StreamingModels([chunk("x")], httpx.ReadTimeout("slow")))

    with pytest.raises(LLMError) as info:
        list(stream(HISTORY))

    assert info.value.status is None
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)


def test_read_error_yields_deltas_then_llm_error_and_no_done(install_models):
    install_models(
        StreamingModels(
            [chunk("Hello "), chunk("world")], httpx.ReadError("connection dropped")
        )
    )

    events = []
    with pytest.raises(LLMError):
        for event in stream(HISTORY):
            events.append(event)  # noqa: PERF402 - spells out the order of events

    assert events == [TextDelta(text="Hello "), TextDelta(text="world")]
    assert not any(isinstance(e, StreamDone) for e in events)
