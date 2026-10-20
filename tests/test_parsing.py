from typing import Literal

import pytest
from pydantic import BaseModel

from ai_lab.parsing import LLMParserError, parse_llm_json


class Summary(BaseModel):
    title: str
    sentiment: Literal["positive", "neutral", "negative"]


@pytest.mark.parametrize(
    ("text", "title", "sentiment"),
    [
        pytest.param(
            '{"title": "x", "sentiment": "positive"}', "x", "positive", id="plain"
        ),
        pytest.param(
            '```json\n{"title": "RAG", "sentiment": "neutral"}\n```',
            "RAG",
            "neutral",
            id="fenced",
        ),
        pytest.param(
            'Sure! Here it is: {"title": "y", "sentiment": "negative"} Anything else?',
            "y",
            "negative",
            id="chatter-before-and-after",
        ),
        pytest.param(
            '{"title": "z", "meta": {"a": 1}, "sentiment": "negative"}',
            "z",
            "negative",
            id="nested-meta",
        ),
    ],
)
def test_valid_inputs(text, title, sentiment):
    result = parse_llm_json(text, Summary)
    assert result == Summary(title=title, sentiment=sentiment)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Sorry, I can't help with that.", id="no-braces"),
        pytest.param('{"title": "x", "sentiment": "happy"}', id="wrong-literal"),
        pytest.param("} oops {", id="reversed-braces"),
        pytest.param('{"title": "x"}', id="missing-field"),
    ],
)
def test_invalid_inputs_raise(text):
    with pytest.raises(LLMParserError):
        parse_llm_json(text, Summary)
