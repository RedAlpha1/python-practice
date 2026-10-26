import json
import logging
from pathlib import Path

import pytest
from google.genai import types
from pydantic import BaseModel

from ai_lab.llm import (
    ChatMessage,
    StreamDone,
    StructuredOutputError,
    ask,
    ask_structured,
    stream,
)
from ai_lab.llm.schemas import LLMResponse
from ai_lab.llm.usage import log_usage_to, record, usage_logger
from tests.llm.fakes import RecordingModels, StreamingModels, chunk, fake_response


def llm_response(model: str = "gemini-3.5-flash-lite") -> LLMResponse:
    return LLMResponse(
        text="secret reply text",
        input_tokens=1_000_000,
        output_tokens=0,
        thinking_tokens=0,
        finish_reason="stop",
        model=model,
        latency_ms=640,
        first_token_ms=None,
    )


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_one_record_is_one_valid_json_line(tmp_path):
    path = tmp_path / "calls.jsonl"
    log_usage_to(path)

    record(llm_response())

    [row] = lines(path)
    assert row["model"] == "gemini-3.5-flash-lite"
    assert (row["input_tokens"], row["output_tokens"], row["thinking_tokens"]) == (
        1_000_000,
        0,
        0,
    )
    assert row["latency_ms"] == 640
    assert row["first_token_ms"] is None
    assert row["finish_reason"] == "stop"
    assert row["cost_usd"] == pytest.approx(0.30)  # 1M input tokens at $0.30
    assert row["ts"].endswith("+00:00")  # an ISO timestamp in UTC


def test_the_reply_text_is_not_logged(tmp_path):
    path = tmp_path / "calls.jsonl"
    log_usage_to(path)

    record(llm_response())

    assert "secret reply text" not in path.read_text()


def test_unknown_model_logs_a_null_cost(tmp_path):
    path = tmp_path / "calls.jsonl"
    log_usage_to(path)

    record(llm_response(model="mystery-model"))

    assert lines(path)[0]["cost_usd"] is None


def test_without_log_usage_to_no_file_is_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    record(llm_response())

    assert list(tmp_path.iterdir()) == []


def test_the_file_is_created_by_the_first_record_not_by_log_usage_to(tmp_path):
    path = tmp_path / "nested" / "calls.jsonl"
    log_usage_to(path)
    assert not path.exists()

    record(llm_response())

    assert path.exists()


def test_calling_log_usage_to_twice_does_not_duplicate_lines(tmp_path):
    path = tmp_path / "calls.jsonl"
    first = log_usage_to(path)
    second = log_usage_to(path)

    record(llm_response())

    assert first is second
    assert len(lines(path)) == 1


def test_records_are_appended_not_overwritten(tmp_path):
    path = tmp_path / "calls.jsonl"
    log_usage_to(path)
    record(llm_response())
    record(llm_response())

    assert len(lines(path)) == 2


def test_root_logger_level_does_not_hide_usage_records(tmp_path):
    path = tmp_path / "calls.jsonl"
    root = logging.getLogger()
    original = root.level
    root.setLevel(logging.WARNING)
    try:
        log_usage_to(path)

        record(llm_response())

        assert len(lines(path)) == 1
        assert usage_logger.level == logging.INFO
    finally:
        root.setLevel(original)


# ── the public calls emit one record each ─────────────────────────────
class Answer(BaseModel):
    ok: bool


def test_ask_logs_one_record(tmp_path, install_models):
    path = tmp_path / "calls.jsonl"
    log_usage_to(path)
    install_models(RecordingModels(fake_response([types.Part(text="hi")])))

    ask("hi", model="m")

    [row] = lines(path)
    assert (row["model"], row["input_tokens"], row["output_tokens"]) == ("m", 11, 4)
    assert row["thinking_tokens"] == 3


def test_stream_logs_one_record_with_first_token_time(tmp_path, install_models):
    path = tmp_path / "calls.jsonl"
    log_usage_to(path)
    install_models(StreamingModels([chunk("Hel"), chunk("lo")]))

    events = list(stream([ChatMessage(role="user", content="hi")], model="m"))

    assert isinstance(events[-1], StreamDone)
    [row] = lines(path)
    assert row["first_token_ms"] is not None
    assert row["latency_ms"] >= row["first_token_ms"]


def test_a_failed_structured_call_is_still_logged(tmp_path, install_models):
    path = tmp_path / "calls.jsonl"
    log_usage_to(path)
    install_models(RecordingModels(fake_response([types.Part(text="not json")])))

    with pytest.raises(StructuredOutputError):
        ask_structured("hi", Answer, model="m")

    [row] = lines(path)  # the tokens were spent even though the output was unusable
    assert row["input_tokens"] == 11
