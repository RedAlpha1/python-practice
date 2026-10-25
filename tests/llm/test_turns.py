import pytest
from google.genai import types

from ai_lab.llm.gemini import _turns_to_contents
from ai_lab.llm.schemas import (
    AssistantTurn,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    UserText,
)


def test_empty_assistant_turn_is_rejected():
    with pytest.raises(ValueError, match="neither text nor tool calls"):
        _turns_to_contents([AssistantTurn()])


def test_assistant_turn_with_only_tool_calls_is_fine():
    call = ToolCall(id="c1", name="get_time", args={"timezone": "Asia/Tokyo"})
    [content] = _turns_to_contents([AssistantTurn(tool_calls=[call])])
    assert content.role == "model"
    assert content.parts and content.parts[0].function_call is not None


def test_provider_raw_is_passed_back_unchanged():
    raw = types.Content(role="model", parts=[types.Part(text="RAW")])
    [content] = _turns_to_contents([AssistantTurn(text="ignored", provider_raw=raw)])
    assert content is raw


def test_roles_and_order():
    contents = _turns_to_contents(
        [
            UserText(text="hi"),
            AssistantTurn(text="hello"),
            ToolResultsTurn(results=[ToolResult(name="add", output=5)]),
        ]
    )
    assert [c.role for c in contents] == ["user", "model", "user"]


def test_tool_result_uses_output_or_error_key():
    ok, bad = (
        _turns_to_contents([ToolResultsTurn(results=[result])])[0].parts
        for result in (
            ToolResult(call_id="c1", name="add", output=5),
            ToolResult(call_id="c2", name="add", error="boom"),
        )
    )
    assert ok and ok[0].function_response is not None
    assert ok[0].function_response.response == {"output": 5}
    assert bad and bad[0].function_response is not None
    assert bad[0].function_response.response == {"error": "boom"}


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(types.Content(role="model", parts=[]), id="empty-parts"),
        pytest.param(types.Content(role="model"), id="no-parts"),
    ],
)
def test_empty_provider_raw_is_rejected_too(raw):
    # A safety block can leave a raw Content with nothing in it; passing it back
    # would bypass the empty-turn check and make Gemini answer 400.
    with pytest.raises(ValueError, match="neither text nor tool calls"):
        _turns_to_contents([AssistantTurn(provider_raw=raw)])
