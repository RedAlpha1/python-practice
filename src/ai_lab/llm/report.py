"""Read the usage JSONL and summarize it per model (calls, tokens, cost, latency)."""

import logging
import math
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class UsageRow(BaseModel):
    """One line of the usage file, as written by ai_lab.llm.usage."""

    ts: str
    model: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    finish_reason: str
    latency_ms: int
    first_token_ms: int | None
    cost_usd: float | None


class ModelStats(BaseModel):
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_usd: float | None  # None when no call of this model had a known price
    avg_ms: float
    p95_ms: int


def load_usage(path: Path) -> list[UsageRow]:
    """Valid lines only: a damaged line is skipped with a warning, never fatal."""
    rows: list[UsageRow] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(UsageRow.model_validate_json(line))
            except ValidationError as e:
                logger.warning("Skipping %s line %d: %s", path, lineno, e.error_count())
    return rows


def p95(values: list[int]) -> int:
    """95% of the values are at or below this (nearest-rank method)."""
    if not values:
        raise ValueError("p95 of an empty list")
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _stats(model: str, rows: list[UsageRow]) -> ModelStats:
    costs = [r.cost_usd for r in rows if r.cost_usd is not None]
    latencies = [r.latency_ms for r in rows]
    return ModelStats(
        model=model,
        calls=len(rows),
        input_tokens=sum(r.input_tokens for r in rows),
        output_tokens=sum(r.output_tokens for r in rows),
        thinking_tokens=sum(r.thinking_tokens for r in rows),
        cost_usd=sum(costs) if costs else None,
        avg_ms=sum(latencies) / len(latencies),
        p95_ms=p95(latencies),
    )


def summarize(rows: list[UsageRow]) -> list[ModelStats]:
    """One ModelStats per model, sorted by model name."""
    by_model: dict[str, list[UsageRow]] = defaultdict(list)
    for row in rows:
        by_model[row.model].append(row)
    return [_stats(model, by_model[model]) for model in sorted(by_model)]


def format_table(rows: list[UsageRow]) -> str:
    """A text table: one line per model, plus a TOTAL line when there are several."""
    stats = summarize(rows)
    if len(stats) > 1:
        stats.append(_stats("TOTAL", rows))

    header = ["model", "calls", "in", "out", "think", "cost($)", "avg ms", "p95 ms"]
    body = [
        [
            s.model,
            f"{s.calls:,}",
            f"{s.input_tokens:,}",
            f"{s.output_tokens:,}",
            f"{s.thinking_tokens:,}",
            "-" if s.cost_usd is None else f"{s.cost_usd:.5f}",
            f"{s.avg_ms:,.0f}",
            f"{s.p95_ms:,}",
        ]
        for s in stats
    ]
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        first, *rest = cells
        numbers = (c.rjust(w) for c, w in zip(rest, widths[1:], strict=True))
        return "  ".join([first.ljust(widths[0]), *numbers])

    return "\n".join([line(header), *(line(r) for r in body)])
