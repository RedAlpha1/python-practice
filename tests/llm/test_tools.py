from datetime import datetime

import pytest
from pydantic import BaseModel

from ai_lab.llm import cost
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


def scripted(*replies: AssistantTurn, model: str = "fake") -> GenerateFn:
    """A fake `generate` that plays the replies in order. No API call.

    Call number n reports latency_ms == n, so a test can tell the rounds apart.
    """
    it = iter(replies)
    n = 0

    def fake(
        turns: list[Turn], tools: list[ToolSpec]
    ) -> tuple[AssistantTurn, LLMResponse]:
        nonlocal n
        n += 1
        return next(it), LLMResponse(
            text="",
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
            finish_reason="stop",
            model=model,
            latency_ms=n,
            first_token_ms=None,
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


def history_of(*args, **kwargs) -> list[Turn]:
    return run_with_tools(*args, **kwargs).history


# ── run_with_tools ────────────────────────────────────────────────────
def test_plain_text_answer_is_two_turns():
    history = history_of(
        QUESTION, [], HANDLERS, generate=scripted(AssistantTurn(text="Hi."))
    )
    assert [t.kind for t in history] == ["user", "assistant"]


def test_one_tool_call_then_text():
    generate = scripted(
        AssistantTurn(tool_calls=[call("add", "c1", a=17, b=25)]),
        AssistantTurn(text="It is 42."),
    )
    history = history_of(QUESTION, [], HANDLERS, generate=generate)

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
    history = history_of(QUESTION, [], HANDLERS, generate=generate)

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
    history = history_of(QUESTION, [], HANDLERS, generate=generate)

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
    history = history_of(QUESTION, [], HANDLERS, generate=generate)

    [result] = last_results(history)
    assert result.error == "ZeroDivisionError: division by zero"


def test_tool_calls_forever_raises_tool_loop_error():
    always = AssistantTurn(tool_calls=[call("add", "c1", a=1, b=1)])
    with pytest.raises(ToolLoopError, match="after 3 steps"):
        history_of(
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
        history_of(QUESTION, [], HANDLERS, generate=generate)
    assert len(seen) == 1  # stopped at once, no second round with a bad turn


def test_callers_turns_list_is_not_mutated():
    turns: list[Turn] = [UserText(text="hi")]
    history = history_of(
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


# ── step 7: the usage of every round ──────────────────────────────────
def test_one_round_for_a_plain_answer():
    result = run_with_tools(
        QUESTION, [], HANDLERS, generate=scripted(AssistantTurn(text="Hi."))
    )
    assert [r.latency_ms for r in result.rounds] == [1]


def test_rounds_match_the_number_of_model_calls_in_order():
    generate = scripted(
        AssistantTurn(tool_calls=[call("add", "c1", a=1, b=2)]),
        AssistantTurn(tool_calls=[call("add", "c2", a=3, b=4)]),
        AssistantTurn(text="Done."),
    )
    result = run_with_tools(QUESTION, [], HANDLERS, generate=generate)

    assert len(result.rounds) == 3
    assert [r.latency_ms for r in result.rounds] == [1, 2, 3]  # in call order


def test_two_calls_in_one_round_are_still_one_model_call():
    generate = scripted(
        AssistantTurn(
            tool_calls=[call("add", "c1", a=1, b=2), call("add", "c2", a=3, b=4)]
        ),
        AssistantTurn(text="Done."),
    )
    result = run_with_tools(QUESTION, [], HANDLERS, generate=generate)

    assert len(result.rounds) == 2  # two API calls, although three turns were added


def test_failed_run_still_reports_the_rounds_it_used():
    always = AssistantTurn(tool_calls=[call("add", "c1", a=1, b=1)])
    with pytest.raises(ToolLoopError) as info:
        run_with_tools(
            QUESTION,
            [],
            HANDLERS,
            generate=scripted(always, always, always),
            max_steps=3,
        )
    assert len(info.value.rounds) == 3


def test_empty_reply_error_includes_the_round_that_was_empty():
    with pytest.raises(ToolLoopError) as info:
        run_with_tools(QUESTION, [], HANDLERS, generate=scripted(AssistantTurn()))
    assert len(info.value.rounds) == 1  # tokens were spent on that empty reply


def test_total_cost_adds_up_the_rounds_of_a_known_model(monkeypatch):
    monkeypatch.setattr(
        cost, "PRICES", {"priced": cost.Price(input_per_m=2.0, output_per_m=10.0)}
    )
    generate = scripted(
        AssistantTurn(tool_calls=[call("add", "c1", a=1, b=2)]),
        AssistantTurn(text="3."),
        model="priced",
    )
    result = run_with_tools(QUESTION, [], HANDLERS, generate=generate)
    # give the two rounds known token counts, then check the sum by hand
    result.rounds[0].input_tokens = 500_000
    result.rounds[1].output_tokens = 100_000

    assert result.total_cost_usd() == pytest.approx(
        500_000 * 2 / 1e6 + 100_000 * 10 / 1e6
    )


def test_total_cost_is_none_if_any_round_is_unpriced():
    result = run_with_tools(
        QUESTION, [], HANDLERS, generate=scripted(AssistantTurn(text="Hi."))
    )  # model "fake" has no price
    assert result.total_cost_usd() is None


def test_two_tool_rounds_then_a_final_answer_make_three_rounds():
    generate = scripted(
        AssistantTurn(tool_calls=[call("add", "c1", a=1, b=2)]),
        AssistantTurn(tool_calls=[call("add", "c2", a=3, b=4)]),
        AssistantTurn(text="Done."),
    )
    result = run_with_tools(QUESTION, [], HANDLERS, generate=generate)

    assert len(result.rounds) == 3


def test_failed_run_error_carries_the_history_so_far():
    always = AssistantTurn(tool_calls=[call("add", "c1", a=1, b=1)])
    with pytest.raises(ToolLoopError) as info:
        run_with_tools(
            QUESTION,
            [],
            HANDLERS,
            generate=scripted(always, always),
            max_steps=2,
        )

    # user, then (assistant, results) twice: everything up to giving up
    assert [t.kind for t in info.value.history] == [
        "user",
        "assistant",
        "tool_results",
        "assistant",
        "tool_results",
    ]


def test_empty_reply_error_history_excludes_the_empty_turn():
    with pytest.raises(ToolLoopError) as info:
        run_with_tools(QUESTION, [], HANDLERS, generate=scripted(AssistantTurn()))

    assert [t.kind for t in info.value.history] == ["user"]


def test_total_tokens_adds_input_output_and_thinking_over_all_rounds():
    generate = scripted(
        AssistantTurn(tool_calls=[call("add", "c1", a=1, b=2)]),
        AssistantTurn(text="3."),
    )
    result = run_with_tools(QUESTION, [], HANDLERS, generate=generate)
    first, second = result.rounds
    first.input_tokens, first.output_tokens, first.thinking_tokens = 10, 5, 100
    second.input_tokens, second.output_tokens, second.thinking_tokens = 20, 7, 0

    assert result.total_tokens == 10 + 5 + 100 + 20 + 7 + 0
