"""Provider-neutral tool loop: ask the model, run the tools it asks for, repeat."""

from collections.abc import Callable

from pydantic_core import PydanticSerializationError, to_jsonable_python

from ai_lab.llm.gemini import generate_turn
from ai_lab.llm.schemas import (
    AssistantTurn,
    LLMResponse,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    ToolSpec,
    Turn,
)

GenerateFn = Callable[[list[Turn], list[ToolSpec]], tuple[AssistantTurn, LLMResponse]]
Handler = Callable[..., object]


class ToolLoopError(Exception):
    """The tool loop could not finish with an answer."""


def run_tool(call: ToolCall, handlers: dict[str, Handler]) -> ToolResult:
    """Run one tool call. Failures become an error result the model can read."""

    def failed(message: str) -> ToolResult:
        return ToolResult(call_id=call.id, name=call.name, error=message)

    if call.name not in handlers:
        return failed(f"unknown tool '{call.name}'. Available: {list(handlers)}")
    try:
        output = handlers[call.name](**call.args)
    except Exception as e:  # noqa: BLE001 - any tool failure goes back to the model
        return failed(f"{type(e).__name__}: {e}")
    try:
        # Pydantic models, datetimes, dataclasses... -> plain JSON-safe data
        output = to_jsonable_python(output)
    except (PydanticSerializationError, ValueError, TypeError) as e:
        return failed(f"tool output is not JSON-serializable: {e}")
    return ToolResult(call_id=call.id, name=call.name, output=output)


def run_with_tools(
    turns: list[Turn],
    tools: list[ToolSpec],
    handlers: dict[str, Handler],
    *,
    generate: GenerateFn = generate_turn,
    max_steps: int = 5,
) -> list[Turn]:
    """Return the full history, ending with the model's final text answer."""
    history = list(turns)  # copy: never mutate the caller's list
    for _ in range(max_steps):
        turn, response = generate(history, tools)

        # An empty turn must never enter the history: Gemini rejects it next round.
        if not turn.text and not turn.tool_calls:
            raise ToolLoopError(f"empty reply, finish_reason={response.finish_reason}")
        history.append(turn)

        if not turn.tool_calls:
            return history  # final answer

        results = [run_tool(call, handlers) for call in turn.tool_calls]
        history.append(ToolResultsTurn(results=results))

    raise ToolLoopError(f"model still asking for tools after {max_steps} steps")
