"""Tools the assistant can call: what time is it, and what did we talk about before."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from ai_lab.llm.schemas import ToolSpec
from ai_lab.llm.tools import Handler
from ai_lab.models import load_conversations

SESSIONS_PATH = Path("data/chat_sessions.jsonl")
MAX_RESULTS = 5  # more would burn tokens and fill the model's context
SNIPPET_CHARS = 120


class GetTimeArgs(BaseModel):
    timezone: str = Field(description="IANA timezone, e.g. Asia/Kolkata, Europe/London")


class SearchSessionsArgs(BaseModel):
    keyword: str = Field(description="Word or phrase to look for in past chats")


def tool_spec(name: str, description: str, args: type[BaseModel]) -> ToolSpec:
    """Build the spec from a Pydantic model, so the JSON Schema is never hand-written."""
    return ToolSpec(
        name=name, description=description, parameter=args.model_json_schema()
    )


def get_time(timezone: str) -> dict[str, str | int]:
    """Structured, so the model never has to parse a string like "03:41"."""
    now = datetime.now(
        ZoneInfo(timezone)
    )  # an unknown name raises: the loop reports it
    return {
        "timezone": timezone,
        "hour": now.hour,
        "minute": now.minute,
        "weekday": now.strftime("%A"),
    }


def search_sessions(keyword: str, path: Path = SESSIONS_PATH) -> list[dict[str, str]]:
    """Messages in saved chats that contain `keyword` (any case), newest chat first."""
    if not keyword.strip():
        raise ValueError("keyword must not be empty")
    if not path.exists():
        return []

    needle = keyword.lower()
    matches: list[dict[str, str]] = []
    for conv in reversed(load_conversations(path)):
        for message in conv.messages:
            if needle in message.content.lower():
                matches.append(
                    {
                        "session_id": conv.id,
                        "started_at": conv.started_at,
                        "snippet": message.content[:SNIPPET_CHARS],
                    }
                )
                if len(matches) == MAX_RESULTS:
                    return matches
    return matches


def _search_sessions_tool(keyword: str) -> list[dict[str, str]]:
    # The model gets this wrapper, not search_sessions itself. search_sessions has a
    # `path` parameter for tests, and the model must never be able to choose a file.
    return search_sessions(keyword)


TOOLS: list[ToolSpec] = [
    tool_spec(
        "get_time",
        "Get the current local time, hour, minute and weekday in a timezone.",
        GetTimeArgs,
    ),
    tool_spec(
        "search_sessions",
        "Search the user's saved past chats for a keyword. Returns up to "
        f"{MAX_RESULTS} matching messages with their session id and date.",
        SearchSessionsArgs,
    ),
]

HANDLERS: dict[str, Handler] = {
    "get_time": get_time,
    "search_sessions": _search_sessions_tool,
}
