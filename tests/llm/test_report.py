import json
import logging
from pathlib import Path

import pytest

from ai_lab.llm.report import UsageRow, format_table, load_usage, p95, summarize


def row(
    model: str = "m1",
    latency_ms: int = 100,
    input_tokens: int = 10,
    output_tokens: int = 5,
    thinking_tokens: int = 0,
    cost_usd: float | None = 0.001,
) -> UsageRow:
    return UsageRow(
        ts="2026-10-26T10:00:00.000+00:00",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        finish_reason="stop",
        latency_ms=latency_ms,
        first_token_ms=None,
        cost_usd=cost_usd,
    )


# ── p95 ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param([7], 7, id="single-value"),
        pytest.param(list(range(1, 101)), 95, id="one-to-hundred"),
        pytest.param(list(range(1, 21)), 19, id="twenty-values-19th"),
        pytest.param([5, 1, 9, 3], 9, id="unsorted-input"),
        pytest.param([100] * 19 + [5000], 100, id="one-slow-call-below-5-percent"),
        pytest.param([100] * 18 + [5000, 6000], 5000, id="two-slow-calls-show-up"),
    ],
)
def test_p95(values, expected):
    assert p95(values) == expected


def test_p95_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        p95([])


def test_average_hides_a_slow_call_but_p95_shows_it():
    rows = [row(latency_ms=100)] * 18 + [row(latency_ms=5000), row(latency_ms=6000)]

    [stats] = summarize(rows)

    assert stats.avg_ms == pytest.approx(640)  # (18*100 + 5000 + 6000) / 20
    assert stats.p95_ms == 5000


# ── summarize ─────────────────────────────────────────────────────────
def test_groups_by_model_and_sums_tokens_and_cost():
    rows = [
        row("b-model", input_tokens=100, output_tokens=10, cost_usd=0.5),
        row(
            "a-model", input_tokens=1, output_tokens=2, thinking_tokens=3, cost_usd=0.1
        ),
        row(
            "a-model", input_tokens=4, output_tokens=5, thinking_tokens=6, cost_usd=0.2
        ),
    ]

    a, b = summarize(rows)

    assert (a.model, a.calls) == ("a-model", 2)  # sorted by name
    assert (a.input_tokens, a.output_tokens, a.thinking_tokens) == (5, 7, 9)
    assert a.cost_usd == pytest.approx(0.3)
    assert (b.model, b.calls, b.cost_usd) == ("b-model", 1, 0.5)


def test_cost_is_none_when_no_price_was_known():
    [stats] = summarize([row(cost_usd=None), row(cost_usd=None)])
    assert stats.cost_usd is None


def test_unknown_prices_are_left_out_of_the_cost_sum():
    [stats] = summarize([row(cost_usd=None), row(cost_usd=0.25)])
    assert stats.cost_usd == 0.25


# ── load_usage ────────────────────────────────────────────────────────
GOOD = {
    "ts": "2026-10-26T10:00:00.000+00:00",
    "model": "m1",
    "input_tokens": 1,
    "output_tokens": 2,
    "thinking_tokens": 0,
    "finish_reason": "stop",
    "latency_ms": 640,
    "first_token_ms": None,
    "cost_usd": None,
}


def test_damaged_lines_are_skipped_with_a_warning(tmp_path: Path, caplog):
    path = tmp_path / "calls.jsonl"
    path.write_text(
        "\n".join(
            [json.dumps(GOOD), "not json", '{"model": "x"}', "", json.dumps(GOOD)]
        )
    )

    with caplog.at_level(logging.WARNING):
        rows = load_usage(path)

    assert len(rows) == 2
    assert "line 2" in caplog.text
    assert "line 3" in caplog.text


# ── format_table ──────────────────────────────────────────────────────
def test_table_has_a_header_and_one_line_per_model():
    rows = [row("only-model", latency_ms=640, input_tokens=3120, output_tokens=890)]

    lines = format_table(rows).splitlines()

    assert len(lines) == 2
    header, line = lines
    for column in (
        "model",
        "calls",
        "in",
        "out",
        "think",
        "cost($)",
        "avg ms",
        "p95 ms",
    ):
        assert column in header
    assert line.split()[:4] == ["only-model", "1", "3,120", "890"]
    assert line.split()[-2:] == ["640", "640"]


def test_table_shows_a_dash_for_unknown_cost():
    assert "-" in format_table([row(cost_usd=None)]).splitlines()[1].split()


def test_table_adds_a_total_line_for_several_models():
    lines = format_table(
        [row("a", input_tokens=1), row("b", input_tokens=2)]
    ).splitlines()

    assert [ln.split()[0] for ln in lines] == ["model", "a", "b", "TOTAL"]
    assert lines[-1].split()[1:3] == ["2", "3"]  # calls, input tokens
