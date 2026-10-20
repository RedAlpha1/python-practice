import httpx
import pytest
from google.genai import errors as genai_errors

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
        pytest.param(_retry_info("abc"), None, id="malformed"),
        pytest.param(_retry_info("nans"), None, id="nan"),
        pytest.param(_retry_info("-5s"), None, id="negative"),
        pytest.param(_retry_info(52), None, id="not-a-string"),
        pytest.param({"error": {"details": ["oops", None]}}, None, id="non-dict-items"),
        pytest.param(
            {"error": {"details": ["oops", {"retryDelay": "7s"}]}},
            7.0,
            id="skips-non-dict-then-finds",
        ),
        pytest.param({"error": {"details": "nope"}}, None, id="details-not-a-list"),
        pytest.param({"error": "nope"}, None, id="error-not-a-dict"),
        pytest.param(None, None, id="details-none"),
        pytest.param([], None, id="details-not-a-dict"),
    ],
)
def test_retry_after(details, expected):
    assert llm._retry_after(details) == expected
