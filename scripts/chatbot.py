import sys
from datetime import datetime
from pathlib import Path

from ai_lab.config import get_settings
from ai_lab.llm import (
    ChatMessage,
    LLMError,
    LLMResponse,
    StreamDone,
    TextDelta,
    log_usage_to,
    stream,
    to_message,
)
from ai_lab.models import Conversation, append_conversation

SYSTEM = "You are a concise assistant. Keep answers under 3 sentences."
SESSIONS_PATH = Path("data/chat_sessions.jsonl")
USAGE_PATH = Path("data/llm_calls.jsonl")
HELP = "Commands: /reset clears the history, /exit saves the chat and quits."

# Optional first argument: thinking budget in tokens (0 turns thinking off).
thinking_budget = int(sys.argv[1]) if len(sys.argv) > 1 else None


def save_session(history: list[ChatMessage], started_at: datetime) -> None:
    if not history:
        return
    session = Conversation(
        id=f"s{started_at:%Y%m%d-%H%M%S}",
        model=get_settings().default_model,
        started_at=started_at.isoformat(timespec="seconds"),
        messages=[to_message(m) for m in history],
    )
    append_conversation(session, SESSIONS_PATH)
    print(f"Saved {len(history)} messages to {SESSIONS_PATH} as {session.id}")


history: list[ChatMessage] = []
turn = 0
started_at = datetime.now()

log_usage_to(USAGE_PATH)
print(f"Chatbot ready. {HELP}")
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
            history, turn, started_at = [], 0, datetime.now()
            print("History cleared.")
            continue
        if not user_text:
            continue

        # Only commit the turn to history once the call succeeds.
        pending = [*history, ChatMessage(role="user", content=user_text)]
        print("bot> ", end="", flush=True)
        result: LLMResponse | None = None
        try:
            for event in stream(
                pending, system=SYSTEM, thinking_budget=thinking_budget
            ):
                match event:
                    case TextDelta(text=text):
                        print(text, end="", flush=True)
                    case StreamDone(response=response):
                        result = response
        except LLMError as e:
            wait = f", retry in {e.retry_after:.0f}s" if e.retry_after else ""
            print(f"\n[stream interrupted; turn not saved] [{e.status}{wait}: {e}]")
            continue
        print()

        if result is None or result.finish_reason != "stop" or not result.text:
            print(
                f"[warning: turn not saved; "
                f"finish_reason={result.finish_reason if result else None}]"
            )
            continue

        history = [*pending, ChatMessage(role="assistant", content=result.text)]
        turn += 1

        print(
            f"     [turn {turn} | messages sent: {len(pending)} | "
            f"in={result.input_tokens} out={result.output_tokens} "
            f"think={result.thinking_tokens} | {result.finish_reason}]"
        )
except KeyboardInterrupt:
    print("\n[interrupted]")
finally:
    save_session(history, started_at)
