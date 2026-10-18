import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass
class Conversation:
    id: str
    model: str
    started_at: str
    messages: list[Message]


def save_conversations(convs: list[Conversation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for conv in convs:
            f.write(json.dumps(asdict(conv)) + "\n")


def load_conversations(path: Path) -> list[Conversation]:
    with open(path) as f:
        return [
            Conversation(
                id=d["id"],
                model=d["model"],
                started_at=d["started_at"],
                messages=[Message(**m) for m in d["messages"]],
            )
            for d in map(json.loads, f)
        ]
