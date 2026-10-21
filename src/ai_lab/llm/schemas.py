"""Provider-neutral types: the contract every LLM backend returns and accepts."""

from typing import Literal

from pydantic import BaseModel

from ai_lab.models import Message

FinishReason = Literal["stop", "max_tokens", "safety", "other"]


class LLMError(Exception):
    """Provider-independent API failure; status and retry_after may be unknown."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class LLMResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    finish_reason: FinishReason


class TextDelta(BaseModel):
    text: str


class StreamDone(BaseModel):
    response: LLMResponse


StreamEvent = TextDelta | StreamDone


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


def to_message(msg: ChatMessage) -> Message:
    return Message(role=msg.role, content=msg.content)


def from_message(msg: Message) -> ChatMessage:
    if msg.role == "system":
        raise ValueError("system messages are not part of a chat history")
    return ChatMessage(role=msg.role, content=msg.content)
