from typing import Literal

import httpx
import pytest
from google.genai import types
from pydantic import BaseModel, Field

from ai_lab.llm import LLMError, StructuredOutputError, ask_structured
from tests.llm.fakes import RecordingModels, fake_response


class Summary(BaseModel):
    topic: str = Field(description="Main topic in 2-5 words")
    sentiment: Literal["positive", "neutral", "negative"]
    action_items: list[str] = []


def reply(text: str, finish=types.FinishReason.STOP):
    return RecordingModels(fake_response([types.Part(text=text)], finish))


def test_valid_json_becomes_the_model_object_plus_usage(install_models):
    install_models(
        reply('{"topic": "RAG basics", "sentiment": "positive", "action_items": ["x"]}')
    )

    summary, usage = ask_structured("summarize", Summary)

    assert summary == Summary(
        topic="RAG basics", sentiment="positive", action_items=["x"]
    )
    assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens) == (
        11,
        4,
        3,
    )


def test_schema_and_json_mode_are_sent_to_the_api(install_models):
    models = reply('{"topic": "t", "sentiment": "neutral"}')
    install_models(models)

    ask_structured("summarize", Summary, system="be brief")

    config = models.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == Summary.model_json_schema()
    assert config.system_instruction == "be brief"
    assert models.kwargs["contents"] == "summarize"


def test_cut_off_json_raises_with_the_raw_text(install_models):
    install_models(reply('{"topic": "RAG', finish=types.FinishReason.MAX_TOKENS))

    with pytest.raises(StructuredOutputError, match="max_tokens") as info:
        ask_structured("summarize", Summary)

    assert info.value.raw == '{"topic": "RAG'


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{"topic": "t", "sentiment": "happy"}', id="wrong-enum-value"),
        pytest.param('{"sentiment": "neutral"}', id="missing-field"),
        pytest.param("Sure! Here is your summary.", id="not-json"),
        pytest.param("", id="empty-reply"),
    ],
)
def test_output_that_does_not_match_raises_with_the_raw_text(install_models, text):
    install_models(reply(text))

    with pytest.raises(StructuredOutputError, match="Summary") as info:
        ask_structured("summarize", Summary)

    assert info.value.raw == text
    assert info.value.__cause__ is not None  # the original ValidationError


def test_thought_parts_are_not_parsed_as_json(install_models):
    parts = [types.Part(text="let me think...", thought=True)]
    parts.append(types.Part(text='{"topic": "t", "sentiment": "neutral"}'))
    install_models(RecordingModels(fake_response(parts)))

    summary, _ = ask_structured("summarize", Summary)

    assert summary.topic == "t"


def test_api_errors_still_become_llm_error(install_models):
    install_models(RecordingModels(error=httpx.ConnectError("dns")))

    with pytest.raises(LLMError):
        ask_structured("summarize", Summary)
