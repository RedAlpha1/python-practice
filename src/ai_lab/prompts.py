from ai_lab.models import Conversation, load_conversations
from textwrap import dedent
from pathlib import Path


def build_summary_prompt(conv: Conversation, max_messages: int = 6) -> str:
    recent = conv.messages[-max_messages:]
    transcript = "\n".join(f"{m.role}: {m.content}" for m in recent)

    header = dedent(f"""\
        You are an assistant that summarizes chat conversations.

        Conversation ID: {conv.id}
        Model: {conv.model}
        Started: {conv.started_at[:10]}
        Messages shown: {len(recent)} of {len(conv.messages)}
    """)

    return f"{header}\n<transcript>\n{transcript}\n</transcript>\n\nSummarize this conversation in 2 sentences."


if __name__ == "__main__":
    convs = load_conversations(Path("data/chats.jsonl"))
    longest = max(convs, key=lambda c: len(c.messages))
    print(build_summary_prompt(longest))
