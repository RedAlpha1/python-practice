from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from ai_lab import llm
from ai_lab.config import get_settings


class _FailingModels:
    def __init__(self, error: Exception):
        self.error = error

    def generate_content(self, **_kwargs):
        raise self.error


class _FakeClient:
    def __init__(self, error: Exception):
        self.models = _FailingModels(error)


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _body(code: int, status: str, details: list[dict] | None = None) -> dict:
    return {
        "error": {
            "code": code,
            "message": "nope",
            "status": status,
            "details": details or [],
        }
    }


def test_rate_limit_becomes_llm_error_with_retry_after(monkeypatch):
    retry_info = {
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": "52s",
    }
    error = genai_errors.ClientError(
        429, _body(429, "RESOURCE_EXHAUSTED", [retry_info])
    )
    monkeypatch.setattr(llm, "_client", lambda: _FakeClient(error))

    with pytest.raises(llm.LLMError) as info:
        llm.ask("hi")

    assert info.value.status == 429
    assert info.value.retry_after == 52.0
    assert isinstance(info.value.__cause__, genai_errors.ClientError)


def test_server_error_has_no_retry_after(monkeypatch):
    error = genai_errors.ServerError(503, _body(503, "UNAVAILABLE"))
    monkeypatch.setattr(llm, "_client", lambda: _FakeClient(error))

    with pytest.raises(llm.LLMError) as info:
        llm.chat([llm.ChatMessage(role="user", content="hi")])

    assert info.value.status == 503
    assert info.value.retry_after is None


def test_roles_round_trip_through_models():
    chat_msg = llm.ChatMessage(role="assistant", content="hello")
    assert llm.from_message(llm.to_message(chat_msg)) == chat_msg


def test_network_failure_becomes_llm_error_without_status(monkeypatch):
    error = httpx.ConnectError("dns failure")
    monkeypatch.setattr(llm, "_client", lambda: _FakeClient(error))

    with pytest.raises(llm.LLMError) as info:
        llm.ask("hi")

    assert info.value.status is None
    assert info.value.retry_after is None
    assert isinstance(info.value.__cause__, httpx.ConnectError)


def _retry_info(delay: object) -> dict:
    return {"error": {"details": [{"retryDelay": delay}]}}


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        pytest.param(_retry_info("52s"), 52.0, id="seconds"),
        pytest.param(_retry_info("52150s"), 52150.0, id="large"),
        pytest.param(_retry_info("1.5s"), 1.5, id="fractional"),
        pytest.param({"error": {"details": [{"other": 1}]}}, None, id="no-delay"),
        pytest.param(_retry_info("abc"), None, id="malformed-no-suffix"),
        pytest.param(_retry_info("abcs"), None, id="malformed-with-suffix"),
        pytest.param(_retry_info("nans"), None, id="nan"),
        pytest.param(_retry_info("-5s"), None, id="negative"),
        pytest.param(_retry_info(52), None, id="not-a-string"),
        pytest.param({"error": {"details": ["oops", None]}}, None, id="non-dict-items"),
        pytest.param(
            {"error": {"details": ["oops", {"retryDelay": "7s"}]}},
            7.0,
            id="skips-non-dict-then-finds",
        ),
        pytest.param(
            {"error": {"details": [{"retryDelay": "abcs"}, {"retryDelay": "7s"}]}},
            7.0,
            id="malformed-then-valid",
        ),
        pytest.param({"error": {"details": "nope"}}, None, id="details-not-a-list"),
        pytest.param({"error": "nope"}, None, id="error-not-a-dict"),
        pytest.param(None, None, id="details-none"),
        pytest.param([], None, id="details-not-a-dict"),
    ],
)
def test_retry_after(details, expected):
    assert llm._retry_after(details) == expected


def _chunk(text=None, usage=None, finish=None):
    candidates = [SimpleNamespace(finish_reason=finish)] if finish else None
    return SimpleNamespace(text=text, usage_metadata=usage, candidates=candidates)


class _StreamingModels:
    """generate_content_stream yields the chunks, then raises `error` if given."""

    def __init__(self, chunks, error: Exception | None = None):
        self.chunks = chunks
        self.error = error

    def generate_content_stream(self, **_kwargs):
        yield from self.chunks
        if self.error:
            raise self.error


def _patch_stream(monkeypatch, chunks, error=None):
    client = SimpleNamespace(models=_StreamingModels(chunks, error))
    monkeypatch.setattr(llm, "_client", lambda: client)


HISTORY = [llm.ChatMessage(role="user", content="hi")]


def test_stream_yields_deltas_then_done(monkeypatch):
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=5, candidates_token_count=7, thoughts_token_count=2
    )
    _patch_stream(
        monkeypatch,
        [
            _chunk("Hel"),
            _chunk(None),  # a thinking chunk has no text: skipped
            _chunk("lo"),
            _chunk("", usage=usage, finish=types.FinishReason.STOP),
        ],
    )

    events = list(llm.stream(HISTORY))

    assert events[:-1] == [llm.TextDelta(text="Hel"), llm.TextDelta(text="lo")]
    done = events[-1]
    assert isinstance(done, llm.StreamDone)
    assert done.response == llm.LLMResponse(
        text="Hello",
        input_tokens=5,
        output_tokens=7,
        thinking_tokens=2,
        finish_reason="stop",
    )


def test_stream_error_mid_stream_after_some_text(monkeypatch):
    error = genai_errors.ServerError(503, _body(503, "UNAVAILABLE"))
    _patch_stream(monkeypatch, [_chunk("par"), _chunk("tial")], error)

    seen = []
    with pytest.raises(llm.LLMError) as info:
        seen.extend(llm.stream(HISTORY))  # keeps what arrived before the error

    assert seen == [llm.TextDelta(text="par"), llm.TextDelta(text="tial")]
    assert info.value.status == 503


def test_stream_network_error_mid_stream(monkeypatch):
    _patch_stream(monkeypatch, [_chunk("x")], httpx.ReadTimeout("slow"))

    with pytest.raises(llm.LLMError) as info:
        list(llm.stream(HISTORY))

    assert info.value.status is None
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)


def test_stream_read_error_yields_deltas_then_llm_error_and_no_done(monkeypatch):
    _patch_stream(
        monkeypatch,
        [_chunk("Hello "), _chunk("world")],
        httpx.ReadError("connection dropped"),
    )

    events = []
    with pytest.raises(llm.LLMError):
        for event in llm.stream(HISTORY):
            events.append(event)  # noqa: PERF402 - spells out the order of events

    assert events == [llm.TextDelta(text="Hello "), llm.TextDelta(text="world")]
    assert not any(isinstance(e, llm.StreamDone) for e in events)
