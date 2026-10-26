"""Provider-neutral tool loop: ask the model, run the tools it asks for, repeat."""

from collections.abc import Callable

from pydantic import BaseModel
from pydantic_core import PydanticSerializationError, to_jsonable_python

from ai_lab.llm.cost import cost_usd
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
    """The tool loop could not finish with an answer.

    `history` and `rounds` hold what happened before it gave up, so a failed run still
    shows how far it got and what it cost.
    """

    def __init__(
        self, message: str, *, history: list[Turn], rounds: list[LLMResponse]
    ) -> None:
        super().__init__(message)
        self.history = history
        self.rounds = rounds


class ToolRunResult(BaseModel):
    history: list[Turn]
    rounds: list[LLMResponse]  # one per model call, in order

    @property
    def total_tokens(self) -> int:
        """Input + output + thinking tokens over all rounds."""
        return sum(
            r.input_tokens + r.output_tokens + r.thinking_tokens for r in self.rounds
        )

    def total_cost_usd(self) -> float | None:
        """Cost of the whole question, or None if any round used an unpriced model."""
        costs = [cost_usd(r) for r in self.rounds]
        if any(c is None for c in costs):
            return None
        return sum(c for c in costs if c is not None)


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
) -> ToolRunResult:
    """Run the loop; the result has the full history and the usage of every round."""
    history = list(turns)  # copy: never mutate the caller's list
    rounds: list[LLMResponse] = []
    for _ in range(max_steps):
        turn, response = generate(history, tools)
        rounds.append(response)

        # An empty turn must never enter the history: Gemini rejects it next round.
        if not turn.text and not turn.tool_calls:
            raise ToolLoopError(
                f"empty reply, finish_reason={response.finish_reason}",
                history=history,
                rounds=rounds,
            )
        history.append(turn)

        if not turn.tool_calls:
            return ToolRunResult(history=history, rounds=rounds)  # final answer

        results = [run_tool(call, handlers) for call in turn.tool_calls]
        history.append(ToolResultsTurn(results=results))

    raise ToolLoopError(
        f"model still asking for tools after {max_steps} steps",
        history=history,
        rounds=rounds,
    )
