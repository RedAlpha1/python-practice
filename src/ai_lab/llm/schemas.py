"""Provider-neutral types: the contract every LLM backend returns and accepts."""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, assert_never

from pydantic import BaseModel, Field, model_validator

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
    model: str
    latency_ms: int  # the API call itself (last attempt), not retry sleeps
    first_token_ms: int | None  # streaming only: time until the first text; else None


class StructuredOutputError(Exception):
    """The model's reply is not usable JSON for our Pydantic model.

    `raw` is the text we got; `response` keeps the usage of the failed call, so its
    cost can still be recorded.
    """

    def __init__(
        self, message: str, *, raw: str, response: LLMResponse | None = None
    ) -> None:
        super().__init__(message)
        self.raw = raw
        self.response = response


class TextDelta(BaseModel):
    text: str


class StreamDone(BaseModel):
    response: LLMResponse


class ToolSpec(BaseModel):
    name: str
    description: str
    parameter: dict[str, Any]


class ToolCall(BaseModel):
    id: str | None = None
    name: str
    args: dict[str, Any] = {}


class ToolResult(BaseModel):
    call_id: str | None = None
    name: str
    output: Any = None
    error: str | None = None

    @model_validator(mode="after")
    def _not_both(self) -> "ToolResult":
        # A result is either a success or a failure, never both.
        if self.error is not None and self.output is not None:
            raise ValueError("ToolResult can't have both output and error")
        return self


class UserText(BaseModel):
    kind: Literal["user"] = "user"
    text: str


class AssistantTurn(BaseModel):
    kind: Literal["assistant"] = "assistant"
    text: str = ""
    tool_calls: list[ToolCall] = []
    provider_raw: Any = Field(default=None, exclude=True)  # sirf gemini.py ke liye


class ToolResultsTurn(BaseModel):
    kind: Literal["tool_results"] = "tool_results"
    results: list[ToolResult]


Turn = Annotated[
    UserText | AssistantTurn | ToolResultsTurn, Field(discriminator="kind")
]
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


def turns_to_messages(turns: Sequence[Turn]) -> list[Message]:
    """The text-only chat to save: user questions and the assistant's final answers.

    Tool plumbing (tool calls, tool results, and any text said while calling a tool)
    does not fit in a Message, so it is left out.
    """
    messages: list[Message] = []
    for turn in turns:
        match turn:
            case UserText(text=text):
                messages.append(Message(role="user", content=text))
            case AssistantTurn(text=text, tool_calls=[]) if text:
                messages.append(Message(role="assistant", content=text))
            case AssistantTurn() | ToolResultsTurn():
                pass
            case _:
                assert_never(turn)
    return messages
