"""Usage records. The library only emits them; the app decides where they go.

Library code must never write files itself: tests would fill the file with fake calls,
and anyone using this package would get our file on their machine. So `record` just
logs to a named logger, and with no handler attached nothing happens. A script opts in
with `log_usage_to(path)`. This is how Python's logging module is meant to be used.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from ai_lab.llm.cost import cost_usd
from ai_lab.llm.schemas import LLMResponse

usage_logger = logging.getLogger("ai_lab.llm.usage")
usage_logger.addHandler(logging.NullHandler())  # the rule for libraries


def record(r: LLMResponse) -> None:
    """Library side: just emit. No handler attached means nothing happens."""
    # The reply text is not usage data (and may be private), so it is left out.
    fields = r.model_dump(exclude={"text"})
    usage_logger.info("llm call", extra={"llm": {**fields, "cost_usd": cost_usd(r)}})


class _JsonLines(logging.Formatter):
    """One JSON object per line: {"ts": ..., "model": ..., "input_tokens": ..., ...}."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).isoformat(
            timespec="milliseconds"
        )
        fields = getattr(record, "llm", None) or {"message": record.getMessage()}
        return json.dumps({"ts": ts, **fields})


def log_usage_to(path: Path) -> logging.Handler:
    """App side: scripts call this once to send usage records to a JSONL file.

    Calling it again for the same file does nothing (no duplicate lines). It returns
    the handler, so a caller can detach it with usage_logger.removeHandler.
    """
    target = path.resolve()
    for existing in usage_logger.handlers:
        if (
            isinstance(existing, logging.FileHandler)
            and Path(existing.baseFilename) == target
        ):
            return existing

    target.parent.mkdir(parents=True, exist_ok=True)
    # delay=True: the file is created by the first record, not by this call
    handler = logging.FileHandler(target, encoding="utf-8", delay=True)
    handler.setFormatter(_JsonLines())
    usage_logger.addHandler(handler)
    usage_logger.setLevel(logging.INFO)  # emit even if the root logger is at WARNING
    return handler
