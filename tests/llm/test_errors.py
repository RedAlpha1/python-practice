import httpx
import pytest
from google.genai import errors as genai_errors

from ai_lab.llm import ChatMessage, LLMError, ask, chat, from_message, to_message
from tests.llm.fakes import FailingModels, error_body


def test_rate_limit_becomes_llm_error_with_retry_after(install_models):
    retry_info = {
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": "52s",
    }
    error = genai_errors.ClientError(
        429, error_body(429, "RESOURCE_EXHAUSTED", [retry_info])
    )
    install_models(FailingModels(error))

    with pytest.raises(LLMError) as info:
        ask("hi")

    assert info.value.status == 429
    assert info.value.retry_after == 52.0
    assert isinstance(info.value.__cause__, genai_errors.ClientError)


def test_server_error_has_no_retry_after(install_models):
    error = genai_errors.ServerError(503, error_body(503, "UNAVAILABLE"))
    install_models(FailingModels(error))

    with pytest.raises(LLMError) as info:
        chat([ChatMessage(role="user", content="hi")])

    assert info.value.status == 503
    assert info.value.retry_after is None


def test_network_failure_becomes_llm_error_without_status(install_models):
    install_models(FailingModels(httpx.ConnectError("dns failure")))

    with pytest.raises(LLMError) as info:
        ask("hi")

    assert info.value.status is None
    assert info.value.retry_after is None
    assert isinstance(info.value.__cause__, httpx.ConnectError)


def test_roles_round_trip_through_models():
    chat_msg = ChatMessage(role="assistant", content="hello")
    assert from_message(to_message(chat_msg)) == chat_msg
