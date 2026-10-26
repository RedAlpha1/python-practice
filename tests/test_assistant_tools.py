from pathlib import Path
from zoneinfo import ZoneInfoNotFoundError

import pytest

from ai_lab.assistant_tools import (
    HANDLERS,
    MAX_RESULTS,
    SNIPPET_CHARS,
    TOOLS,
    GetTimeArgs,
    get_time,
    search_sessions,
    tool_spec,
)
from ai_lab.llm.schemas import ToolCall
from ai_lab.llm.tools import run_tool
from ai_lab.models import Conversation, Message, append_conversation

WEEKDAYS = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}


def session(
    id: str, *texts: str, started_at: str = "2026-10-20T10:00:00"
) -> Conversation:
    roles = ("user", "assistant")
    return Conversation(
        id=id,
        model="m",
        started_at=started_at,
        messages=[
            Message(role=roles[i % 2], content=text) for i, text in enumerate(texts)
        ],
    )


@pytest.fixture
def sessions_file(tmp_path: Path) -> Path:
    return tmp_path / "sessions.jsonl"


# ── get_time ──────────────────────────────────────────────────────────
def test_get_time_returns_structured_data():
    result = get_time("Asia/Tokyo")

    assert set(result) == {"timezone", "hour", "minute", "weekday"}
    assert result["timezone"] == "Asia/Tokyo"
    assert isinstance(result["hour"], int)
    assert isinstance(result["minute"], int)
    assert 0 <= result["hour"] <= 23
    assert 0 <= result["minute"] <= 59
    assert result["weekday"] in WEEKDAYS


def test_get_time_rejects_a_city_name_that_is_not_an_iana_zone():
    with pytest.raises(ZoneInfoNotFoundError):
        get_time("Mumbai")


def test_the_loop_turns_that_exception_into_an_error_result_for_the_model():
    call = ToolCall(id="c1", name="get_time", args={"timezone": "Mumbai"})

    result = run_tool(call, HANDLERS)

    assert result.output is None
    assert result.error is not None
    assert result.error.startswith("ZoneInfoNotFoundError")


# ── search_sessions ───────────────────────────────────────────────────
def test_finds_the_matching_session_only(sessions_file):
    append_conversation(
        session("s-rag", "Tell me about RAG", "RAG fetches documents"), sessions_file
    )
    append_conversation(
        session("s-other", "What is a dataclass?", "A class decorator"), sessions_file
    )

    results = search_sessions("RAG", sessions_file)

    assert {r["session_id"] for r in results} == {"s-rag"}
    assert len(results) == 2  # one per matching message
    assert set(results[0]) == {"session_id", "started_at", "snippet"}


def test_match_is_case_insensitive(sessions_file):
    append_conversation(session("s1", "I like Streaming responses"), sessions_file)

    assert len(search_sessions("streaming", sessions_file)) == 1
    assert len(search_sessions("STREAMING", sessions_file)) == 1


def test_snippet_is_the_first_120_characters_of_the_message(sessions_file):
    long_text = "needle " + "x" * 300
    append_conversation(session("s1", long_text), sessions_file)

    [result] = search_sessions("needle", sessions_file)

    assert result["snippet"] == long_text[:SNIPPET_CHARS]
    assert len(result["snippet"]) == 120


def test_at_most_five_results(sessions_file):
    texts = [f"match number {i}" for i in range(10)]
    append_conversation(session("s1", *texts), sessions_file)

    results = search_sessions("match", sessions_file)

    assert len(results) == MAX_RESULTS == 5


def test_newest_session_comes_first_when_results_are_capped(sessions_file):
    for n in range(7):
        append_conversation(session(f"s{n}", "match"), sessions_file)

    results = search_sessions("match", sessions_file)

    assert [r["session_id"] for r in results] == ["s6", "s5", "s4", "s3", "s2"]


def test_missing_file_gives_an_empty_list_not_a_crash(tmp_path):
    assert search_sessions("anything", tmp_path / "nope.jsonl") == []


def test_no_match_gives_an_empty_list(sessions_file):
    append_conversation(session("s1", "hello"), sessions_file)

    assert search_sessions("zebra", sessions_file) == []


@pytest.mark.parametrize("keyword", ["", "   "])
def test_empty_keyword_is_an_error_not_a_match_for_everything(sessions_file, keyword):
    append_conversation(session("s1", "hello"), sessions_file)

    with pytest.raises(ValueError, match="keyword"):
        search_sessions(keyword, sessions_file)


# ── the tool specs ────────────────────────────────────────────────────
def test_every_spec_has_a_handler_and_the_other_way_round():
    assert {t.name for t in TOOLS} == set(HANDLERS)


def test_spec_schema_comes_from_the_pydantic_model():
    spec = tool_spec("get_time", "desc", GetTimeArgs)

    assert spec.parameter["required"] == ["timezone"]
    assert "IANA" in spec.parameter["properties"]["timezone"]["description"]


def test_the_model_cannot_choose_which_file_search_reads():
    handler = HANDLERS["search_sessions"]

    with pytest.raises(TypeError):
        handler(keyword="x", path="/etc/passwd")
