from datetime import datetime

import pytest
from pydantic import BaseModel

from ai_lab.llm.schemas import (
    AssistantTurn,
    LLMResponse,
    ToolCall,
    ToolResultsTurn,
    ToolSpec,
    Turn,
    UserText,
)
from ai_lab.llm.tools import GenerateFn, ToolLoopError, run_tool, run_with_tools


def scripted(*replies: AssistantTurn) -> GenerateFn:
    """A fake `generate` that plays the replies in order. No API call."""
    it = iter(replies)

    def fake(
        turns: list[Turn], tools: list[ToolSpec]
    ) -> tuple[AssistantTurn, LLMResponse]:
        return next(it), LLMResponse(
            text="",
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
            finish_reason="stop",
        )

    return fake


def add(a: int, b: int) -> int:
    return a + b


def boom(**_kwargs) -> int:
    raise ZeroDivisionError("division by zero")


HANDLERS = {"add": add, "boom": boom}
QUESTION: list[Turn] = [UserText(text="17 + 25?")]


def call(name: str, id: str, **args) -> ToolCall:
    return ToolCall(id=id, name=name, args=args)


def last_results(history: list[Turn]):
    turn = history[-2]
    assert isinstance(turn, ToolResultsTurn)
    return turn.results


# ── run_with_tools ────────────────────────────────────────────────────
def test_plain_text_answer_is_two_turns():
    history = run_with_tools(
        QUESTION, [], HANDLERS, generate=scripted(AssistantTurn(text="Hi."))
    )
    assert [t.kind for t in history] == ["user", "assistant"]


def test_one_tool_call_then_text():
    generate = scripted(
        AssistantTurn(tool_calls=[call("add", "c1", a=17, b=25)]),
        AssistantTurn(text="It is 42."),
    )
    history = run_with_tools(QUESTION, [], HANDLERS, generate=generate)

    assert [t.kind for t in history] == [
        "user",
        "assistant",
        "tool_results",
        "assistant",
    ]
    [result] = last_results(history)
    assert (result.call_id, result.name, result.output) == ("c1", "add", 42)


def test_two_calls_in_one_round_make_one_results_turn():
    generate = scripted(
        AssistantTurn(
            tool_calls=[call("add", "c1", a=1, b=2), call("add", "c2", a=10, b=20)]
        ),
        AssistantTurn(text="3 and 30."),
    )
    history = run_with_tools(QUESTION, [], HANDLERS, generate=generate)

    results_turns = [t for t in history if isinstance(t, ToolResultsTurn)]
    assert len(results_turns) == 1
    assert [(r.call_id, r.output) for r in results_turns[0].results] == [
        ("c1", 3),
        ("c2", 30),
    ]


def test_unknown_tool_becomes_error_result_not_crash():
    generate = scripted(
        AssistantTurn(tool_calls=[call("get_weather", "c1", city="Pune")]),
        AssistantTurn(text="I have no weather tool."),
    )
    history = run_with_tools(QUESTION, [], HANDLERS, generate=generate)

    [result] = last_results(history)
    assert result.output is None
    assert result.error is not None
    assert "unknown tool 'get_weather'" in result.error
    assert "Available: ['add', 'boom']" in result.error


def test_handler_exception_becomes_error_result():
    generate = scripted(
        AssistantTurn(tool_calls=[call("boom", "c1")]),
        AssistantTurn(text="That failed."),
    )
    history = run_with_tools(QUESTION, [], HANDLERS, generate=generate)

    [result] = last_results(history)
    assert result.error == "ZeroDivisionError: division by zero"


def test_tool_calls_forever_raises_tool_loop_error():
    always = AssistantTurn(tool_calls=[call("add", "c1", a=1, b=1)])
    with pytest.raises(ToolLoopError, match="after 3 steps"):
        run_with_tools(
            QUESTION,
            [],
            HANDLERS,
            generate=scripted(always, always, always),
            max_steps=3,
        )


def test_empty_reply_raises_and_is_not_added_to_history():
    seen: list[list[Turn]] = []

    def generate(turns: list[Turn], tools: list[ToolSpec]):
        seen.append(list(turns))
        return scripted(AssistantTurn())(turns, tools)

    with pytest.raises(ToolLoopError, match="empty reply, finish_reason=stop"):
        run_with_tools(QUESTION, [], HANDLERS, generate=generate)
    assert len(seen) == 1  # stopped at once, no second round with a bad turn


def test_callers_turns_list_is_not_mutated():
    turns: list[Turn] = [UserText(text="hi")]
    history = run_with_tools(
        turns, [], HANDLERS, generate=scripted(AssistantTurn(text="Hello."))
    )
    assert len(turns) == 1
    assert len(history) == 2
    assert history is not turns


# ── run_tool ──────────────────────────────────────────────────────────
class Point(BaseModel):
    x: int
    y: int


def test_pydantic_output_is_converted_to_json_data():
    result = run_tool(call("origin", "c1"), {"origin": lambda: Point(x=0, y=0)})
    assert result.output == {"x": 0, "y": 0}


def test_datetime_output_is_converted_to_json_data():
    when = datetime(2026, 10, 26, 9, 30)
    result = run_tool(call("now", "c1"), {"now": lambda: when})
    assert result.output == "2026-10-26T09:30:00"


def test_unserializable_output_becomes_error_result():
    result = run_tool(call("odd", "c1"), {"odd": lambda: object()})
    assert result.output is None
    assert result.error is not None
    assert "not JSON-serializable" in result.error
