from dataclasses import asdict, dataclass
import json
from pathlib import Path
from sys import exception


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass
class Conversation:
    model: str
    messages: list[Message]
    id: str


# Write save_conversations(convs: list[Conversation], path: Path) -> None.
# It writes one conversation per line to data/convs.jsonl.
# Hint: from dataclasses import asdict, and asdict(conv) converts the whole nested object back to a dict.


def save_conversations(convs: list[Conversation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for conv in convs:
            f.write(json.dumps(asdict(conv)) + "\n")


original = [
    Conversation(
        id="c1",
        model="claude",
        messages=[Message("user", "hi"), Message("assistant", "hello!")],
    ),
    Conversation(id="c2", model="gpt", messages=[Message("user", "explain RAG")]),
]
# save_conversations(original, Path("data/convs.jsonl"))


def load_conversations(path: Path) -> list[Conversation]:
    convs: list[Conversation] = []
    try:
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Ignore this fucking line {line_no}")
                    continue
                messages = [Message(**m) for m in raw["messages"]]
                convs.append(Conversation(**{**raw, "messages": messages}))
    except FileNotFoundError:
        print(f"Warning {path} not found")
        return []
    return convs


loaded = load_conversations(Path("data/convs.jsonl"))
print(loaded == original)
