from collections import Counter
from pathlib import Path

from ai_lab.models import Conversation, load_conversations


def total_messages_per_model(convs: list[Conversation]) -> dict[str, int]:
    unique_models = {c.model for c in convs}
    return {
        model: sum(len(c.messages) for c in convs if c.model == model)
        for model in sorted(unique_models)
    }


def avg_turns(convs: list[Conversation]) -> float:
    if not convs:
        return 0.0
    total_messages = sum(len(c.messages) for c in convs)
    return total_messages / len(convs)


def longest_conversation(convs: list[Conversation]) -> Conversation | None:
    if not convs:
        return None
    return max(convs, key=lambda c: len(c.messages))


def busiest_day(convs: list[Conversation]) -> str | None:
    if not convs:
        return None
    days = Counter(c.started_at[:10] for c in convs)
    day, _count = days.most_common(1)[0]
    return day


if __name__ == "__main__":
    convs = load_conversations(Path("data/chats.jsonl"))

    print("Messages per model:", total_messages_per_model(convs))
    print(f"Average turns: {avg_turns(convs):.2f}")

    longest = longest_conversation(convs)
    if longest:
        print(f"Longest chat: {longest.id} ({len(longest.messages)} messages)")

    print("Busiest day:", busiest_day(convs))
