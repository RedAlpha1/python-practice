from ai_lab.models import Conversation, Message
from ai_lab.stats import (
    avg_turns,
    busiest_day,
    longest_conversation,
    total_messages_per_model,
)


def make_conv(
    id: str,
    model: str,
    n_messages: int,
    started_at: str = "2026-10-01T10:00:00",
) -> Conversation:
    msgs = [
        Message("user" if i % 2 == 0 else "assistant", "x") for i in range(n_messages)
    ]
    return Conversation(id=id, model=model, started_at=started_at, messages=msgs)


def test_avg_turns():
    convs = [make_conv("a", "gpt", 2), make_conv("b", "gpt", 4)]
    assert avg_turns(convs) == 3.0


def test_avg_turns_empty():
    assert avg_turns([]) == 0.0


def test_total_messages_per_model():
    convs = [
        make_conv("a", "gpt", 2),
        make_conv("b", "gpt", 4),
        make_conv("c", "claude", 3),
    ]
    assert total_messages_per_model(convs) == {"claude": 3, "gpt": 6}


def test_longest_conversation():
    convs = [make_conv("a", "gpt", 2), make_conv("b", "gpt", 5), make_conv("c", "gpt", 3)]
    longest = longest_conversation(convs)
    assert longest is not None
    assert longest.id == "b"


def test_longest_conversation_empty():
    assert longest_conversation([]) is None


def test_busiest_day():
    convs = [
        make_conv("a", "gpt", 2, started_at="2026-10-05T09:00:00"),
        make_conv("b", "gpt", 2, started_at="2026-10-05T18:30:00"),
        make_conv("c", "gpt", 2, started_at="2026-10-01T10:00:00"),
    ]
    assert busiest_day(convs) == "2026-10-05"
