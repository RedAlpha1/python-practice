"""Public LLM API. Import from here, not from the backend modules."""

from ai_lab.llm.gemini import ask, ask_structured, chat, stream
from ai_lab.llm.schemas import (
    ChatMessage,
    FinishReason,
    LLMError,
    LLMResponse,
    StreamDone,
    StreamEvent,
    StructuredOutputError,
    TextDelta,
    from_message,
    to_message,
)
from ai_lab.llm.usage import log_usage_to

__all__ = [
    "ChatMessage",
    "FinishReason",
    "LLMError",
    "LLMResponse",
    "StreamDone",
    "StreamEvent",
    "StructuredOutputError",
    "TextDelta",
    "ask",
    "ask_structured",
    "chat",
    "from_message",
    "log_usage_to",
    "stream",
    "to_message",
]
