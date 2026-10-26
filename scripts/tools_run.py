import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_lab.llm import log_usage_to
from ai_lab.llm.schemas import AssistantTurn, ToolSpec, Turn, UserText
from ai_lab.llm.tools import ToolLoopError, run_with_tools


def get_time(timezone: str) -> str:
    return datetime.now(ZoneInfo(timezone)).strftime("%H:%M")


def add(a: int, b: int) -> int:
    return a + b


TOOLS = [
    ToolSpec(
        name="get_time",
        description="Current time in an IANA timezone, e.g. Asia/Tokyo",
        parameter={
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
    ),
    ToolSpec(
        name="add",
        description="Add two integers",
        parameter={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    ),
]
HANDLERS = {"get_time": get_time, "add": add}

log_usage_to(Path("data/llm_calls.jsonl"))

question = (
    sys.argv[1]
    if len(sys.argv) > 1
    else ("What time is it in Tokyo and in London? Then add 5 to the Tokyo hour.")
)
turns: list[Turn] = [UserText(text=question)]

try:
    result = run_with_tools(turns, TOOLS, HANDLERS)
except ToolLoopError as e:
    raise SystemExit(
        f"Tool loop failed: {e} (model calls made: {len(e.rounds)})"
    ) from e

final = result.history[-1]
assert isinstance(final, AssistantTurn)
print(final.text)

cost = result.total_cost_usd()
print(
    f"\n{len(result.rounds)} model calls | "
    f"in={sum(r.input_tokens for r in result.rounds)} "
    f"out={sum(r.output_tokens for r in result.rounds)} "
    f"think={sum(r.thinking_tokens for r in result.rounds)} | "
    f"{sum(r.latency_ms for r in result.rounds)} ms | "
    f"cost={'unknown (no price for this model)' if cost is None else f'${cost:.6f}'}"
)
