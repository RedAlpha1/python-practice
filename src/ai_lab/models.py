import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class Conversation(BaseModel):
    id: str
    model: str
    started_at: str
    messages: list[Message] = Field(min_length=1)


def save_conversations(convs: list[Conversation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for conv in convs:
            f.write(conv.model_dump_json() + "\n")


def load_conversations(path: Path) -> list[Conversation]:
    convs = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                convs.append(Conversation.model_validate_json(line))
            except ValidationError as e:
                problems = "; ".join(
                    f"{'.'.join(map(str, err['loc']))}: {err['msg']}"
                    for err in e.errors()
                )
                logger.warning("Skipping %s line %d: %s", path, lineno, problems)
    return convs
