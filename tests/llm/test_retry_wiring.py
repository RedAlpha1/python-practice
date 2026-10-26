"""The retry decorator is applied to the public calls, and not to stream."""

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from ai_lab.llm import ChatMessage, LLMError, ask, chat, stream
from tests.llm.fakes import StreamingModels, chunk, error_body, fake_response


class FailThenSucceed:
    """generate_content raises the errors one per call, then returns a text answer."""

    def __init__(self, *errors: Exception):
        self.errors = list(errors)
        self.calls = 0

    def generate_content(self, **_kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return fake_response([types.Part(text="hello")])


def test_ask_retries_a_503_and_then_succeeds(install_models):
    models = FailThenSucceed(
        genai_errors.ServerError(503, error_body(503, "UNAVAILABLE"))
    )
    install_models(models)

    assert ask("hi").text == "hello"
    assert models.calls == 2


def test_chat_retries_a_network_error(install_models):
    models = FailThenSucceed(httpx.ConnectError("dns"))
    install_models(models)

    result = chat([ChatMessage(role="user", content="hi")])

    assert result.text == "hello"
    assert models.calls == 2


def test_ask_does_not_retry_a_bad_request(install_models):
    models = FailThenSucceed(
        genai_errors.ClientError(400, error_body(400, "INVALID_ARGUMENT"))
    )
    install_models(models)

    with pytest.raises(LLMError) as info:
        ask("hi")

    assert info.value.status == 400
    assert models.calls == 1


def test_stream_is_not_retried(install_models):
    calls = 0

    class CountingStream(StreamingModels):
        def generate_content_stream(self, **kwargs):
            nonlocal calls
            calls += 1
            return super().generate_content_stream(**kwargs)

    error = genai_errors.ServerError(503, error_body(503, "UNAVAILABLE"))
    install_models(CountingStream([chunk("partial ")], error))

    with pytest.raises(LLMError):
        list(stream([ChatMessage(role="user", content="hi")]))

    assert calls == 1  # a retry would replay text the user has already seen
