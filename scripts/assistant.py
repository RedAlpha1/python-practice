import functools
from datetime import datetime
from pathlib import Path

from ai_lab.assistant_tools import HANDLERS, SESSIONS_PATH, TOOLS
from ai_lab.config import get_settings
from ai_lab.llm import LLMError, log_usage_to
from ai_lab.llm.gemini import generate_turn
from ai_lab.llm.schemas import (
    AssistantTurn,
    ToolResultsTurn,
    Turn,
    UserText,
    turns_to_messages,
)
from ai_lab.llm.tools import ToolLoopError, ToolRunResult, run_with_tools
from ai_lab.models import Conversation, append_conversation

SYSTEM = (
    "You are a helpful assistant. Use get_time for any question about the current time. "
    "Use search_sessions when the user asks about their past chats. Keep answers short."
)
USAGE_PATH = Path("data/llm_calls.jsonl")
HELP = "Commands: /reset clears the history, /exit saves the chat and quits."


def save_session(history: list[Turn], started_at: datetime) -> None:
    messages = turns_to_messages(history)  # only questions and final answers
    if not messages:
        return
    session = Conversation(
        id=f"a{started_at:%Y%m%d-%H%M%S}",
        model=get_settings().default_model,
        started_at=started_at.isoformat(timespec="seconds"),
        messages=messages,
    )
    append_conversation(session, SESSIONS_PATH)
    print(f"Saved {len(messages)} messages to {SESSIONS_PATH} as {session.id}")


def usage_line(run: ToolRunResult) -> str:
    cost = run.total_cost_usd()
    cost_text = "cost n/a" if cost is None else f"${cost:.5f}"
    return f"[rounds {len(run.rounds)} | tokens {run.total_tokens:,} | {cost_text}]"


def show_tool_activity(new_turns: list[Turn]) -> None:
    """Print one line per tool the model called, and any tool that failed."""
    for turn in new_turns:
        match turn:
            case AssistantTurn(tool_calls=calls):
                for call in calls:
                    args = ", ".join(f"{k}={v}" for k, v in call.args.items())
                    print(f"🔧 {call.name}({args})")
            case ToolResultsTurn(results=results):
                for r in results:
                    if r.error is not None:
                        print(f"   ↳ {r.name} failed: {r.error}")
            case _:
                pass


def main() -> None:
    log_usage_to(USAGE_PATH)
    generate = functools.partial(generate_turn, system=SYSTEM)
    history: list[Turn] = []
    started_at = datetime.now()

    print(f"Assistant ready. {HELP}")
    try:
        while True:
            try:
                user_text = input("you> ").strip()
            except EOFError:
                break
            if user_text == "/exit":
                break
            if user_text == "/reset":
                save_session(history, started_at)
                history, started_at = [], datetime.now()
                print("History cleared.")
                continue
            if not user_text:
                continue

            # Only commit the turn to the history once the whole run succeeds.
            pending: list[Turn] = [*history, UserText(text=user_text)]
            try:
                result = run_with_tools(pending, TOOLS, HANDLERS, generate=generate)
            except ToolLoopError as e:
                spent = ToolRunResult(history=e.history, rounds=e.rounds)
                show_tool_activity(e.history[len(pending) :])
                print(f"[tool loop failed: {e}; turn not saved] {usage_line(spent)}")
                continue
            except LLMError as e:
                wait = f", retry in {e.retry_after:.0f}s" if e.retry_after else ""
                print(f"[API error {e.status}{wait}; turn not saved: {e}]")
                continue

            show_tool_activity(result.history[len(pending) :])
            final = result.history[-1]
            assert isinstance(final, AssistantTurn)
            print(f"bot> {final.text}")
            print(f"     {usage_line(result)}")
            history = result.history
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        save_session(history, started_at)


if __name__ == "__main__":
    main()
