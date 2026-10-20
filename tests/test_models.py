from pathlib import Path

from ai_lab.models import (
    Conversation,
    Message,
    append_conversation,
    load_conversations,
)


def make_conv(id: str) -> Conversation:
    return Conversation(
        id=id,
        model="gpt",
        started_at="2026-10-01T10:00:00",
        messages=[Message(role="user", content="hi")],
    )


def test_append_creates_file_and_parent_dir(tmp_path: Path):
    path = tmp_path / "nested" / "sessions.jsonl"
    append_conversation(make_conv("a"), path)
    assert [c.id for c in load_conversations(path)] == ["a"]


def test_append_keeps_existing_conversations(tmp_path: Path):
    path = tmp_path / "sessions.jsonl"
    append_conversation(make_conv("a"), path)
    before = path.read_text()
    append_conversation(make_conv("b"), path)

    assert path.read_text().startswith(before)  # old lines untouched
    assert [c.id for c in load_conversations(path)] == ["a", "b"]
