from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ai_lab.llm import LLMError, StructuredOutputError, ask_structured, log_usage_to
from ai_lab.models import load_conversations
from ai_lab.prompts import build_summary_prompt

SESSIONS_PATH = Path("data/chat_sessions.jsonl")
USAGE_PATH = Path("data/llm_calls.jsonl")


class Summary(BaseModel):
    topic: str = Field(description="Main topic in 2-5 words")
    sentiment: Literal["positive", "neutral", "negative"]
    action_items: list[str] = Field(description="Concrete next steps, may be empty")


def main() -> None:
    log_usage_to(USAGE_PATH)
    sessions = load_conversations(SESSIONS_PATH) if SESSIONS_PATH.exists() else []
    if not sessions:
        raise SystemExit(f"No saved sessions in {SESSIONS_PATH}. Run chatbot.py first.")
    conv = sessions[-1]  # the latest session
    print(f"Summarizing {conv.id} ({len(conv.messages)} messages)")

    try:
        summary, usage = ask_structured(build_summary_prompt(conv), Summary)
    except StructuredOutputError as e:
        raise SystemExit(f"Bad model output: {e}\nRaw text: {e.raw!r}") from e
    except LLMError as e:
        raise SystemExit(f"API error {e.status}: {e}") from e

    print(f"Topic      : {summary.topic}")
    print(f"Sentiment  : {summary.sentiment}")
    print("Action items:" + ("" if summary.action_items else " (none)"))
    for item in summary.action_items:
        print(f"  - {item}")
    print(
        f"Tokens     : in={usage.input_tokens} out={usage.output_tokens} "
        f"think={usage.thinking_tokens} ({usage.finish_reason})"
    )


if __name__ == "__main__":
    main()
