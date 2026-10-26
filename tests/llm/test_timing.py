"""latency_ms, first_token_ms and model on every LLMResponse, with a fake clock."""

import itertools
import time

import pytest
from google.genai import types
from pydantic import BaseModel

from ai_lab.llm import ChatMessage, ask, ask_structured, chat, stream
from ai_lab.llm.gemini import generate_turn
from ai_lab.llm.schemas import StreamDone, UserText
from tests.llm.fakes import RecordingModels, StreamingModels, chunk, fake_response


@pytest.fixture
def clock(monkeypatch):
    """clock(0.0, 0.25): successive time.perf_counter() readings; the last repeats."""

    def install(*readings: float) -> None:
        values = itertools.chain(readings, itertools.repeat(readings[-1]))
        monkeypatch.setattr(time, "perf_counter", lambda: next(values))

    return install


class Answer(BaseModel):
    ok: bool


def text_reply(text: str = "hi") -> RecordingModels:
    return RecordingModels(fake_response([types.Part(text=text)]))


def test_ask_records_model_and_latency(install_models, clock):
    install_models(text_reply())
    clock(0.0, 0.25)  # the API call took 0.25 s

    result = ask("hi", model="test-model")

    assert result.model == "test-model"
    assert result.latency_ms == 250
    assert result.first_token_ms is None  # only streaming has a first token


def test_chat_records_latency(install_models, clock):
    install_models(text_reply())
    clock(10.0, 10.5)

    result = chat([ChatMessage(role="user", content="hi")], model="m")

    assert result.latency_ms == 500


def test_generate_turn_records_latency(install_models, clock):
    install_models(text_reply())
    clock(1.0, 1.125)

    _, result = generate_turn([UserText(text="hi")], [], model="m")

    assert (result.model, result.latency_ms) == ("m", 125)


def test_ask_structured_records_latency(install_models, clock):
    install_models(text_reply('{"ok": true}'))
    clock(0.0, 0.75)

    _, result = ask_structured("hi", Answer, model="m")

    assert (result.model, result.latency_ms) == ("m", 750)


def test_default_model_name_comes_from_settings(install_models, monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL", "from-env")
    install_models(text_reply())

    assert ask("hi").model == "from-env"


def done_of(events) -> StreamDone:
    done = events[-1]
    assert isinstance(done, StreamDone)
    return done


def test_stream_has_time_to_first_token_and_total_time(install_models, clock):
    install_models(StreamingModels([chunk("Hel"), chunk("lo"), chunk("")]))
    clock(0.0, 0.1, 0.4)  # start, first word seen, stream finished

    done = done_of(list(stream([ChatMessage(role="user", content="hi")], model="m")))

    assert done.response.first_token_ms == 100
    assert done.response.latency_ms == 400
    assert done.response.model == "m"


def test_first_token_ignores_chunks_without_text(install_models, clock):
    # A thinking chunk arrives first with no text: the user has seen nothing yet.
    install_models(StreamingModels([chunk(None), chunk("Hi")]))
    clock(0.0, 0.3, 0.5)

    done = done_of(list(stream([ChatMessage(role="user", content="hi")])))

    assert done.response.first_token_ms == 300


def test_stream_without_any_text_has_no_first_token(install_models, clock):
    install_models(StreamingModels([chunk(None), chunk("")]))
    clock(0.0, 0.2)

    done = done_of(list(stream([ChatMessage(role="user", content="hi")])))

    assert done.response.first_token_ms is None
    assert done.response.latency_ms == 200
