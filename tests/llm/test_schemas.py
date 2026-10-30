import pytest
from pydantic import ValidationError

from ai_lab.llm.schemas import (
    AssistantTurn,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    UserText,
    turns_to_messages,
)


def test_tool_result_rejects_output_and_error_together():
    with pytest.raises(ValidationError, match="both output and error"):
        ToolResult(name="add", output=5, error="failed")


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"output": 5}, id="success"),
        pytest.param({"error": "boom"}, id="failure"),
        pytest.param({"error": ""}, id="empty-error-is-still-an-error"),
        pytest.param({"output": 0}, id="falsy-output-is-still-output"),
        pytest.param({}, id="tool-returned-nothing"),
    ],
)
def test_tool_result_allows_one_or_neither(fields):
    ToolResult(name="add", **fields)


def test_saved_chat_keeps_questions_and_final_answers_only():
    call = ToolCall(id="c1", name="get_time", args={"timezone": "Asia/Tokyo"})
    turns = [
        UserText(text="time in Tokyo?"),
        AssistantTurn(tool_calls=[call]),
        ToolResultsTurn(results=[ToolResult(call_id="c1", name="get_time", output={})]),
        AssistantTurn(text="It is 03:41."),
        UserText(text="thanks"),
        AssistantTurn(text="You're welcome."),
    ]

    messages = turns_to_messages(turns)

    assert [(m.role, m.content) for m in messages] == [
        ("user", "time in Tokyo?"),
        ("assistant", "It is 03:41."),
        ("user", "thanks"),
        ("assistant", "You're welcome."),
    ]


def test_text_said_while_calling_a_tool_is_not_a_final_answer():
    call = ToolCall(id="c1", name="get_time", args={"timezone": "Asia/Tokyo"})
    turns = [
        UserText(text="time?"),
        AssistantTurn(text="Let me check.", tool_calls=[call]),
        AssistantTurn(text="03:41."),
    ]

    assert [m.content for m in turns_to_messages(turns)] == ["time?", "03:41."]


def test_no_turns_no_messages():
    assert turns_to_messages([]) == []
