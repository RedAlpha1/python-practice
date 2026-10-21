"""Plain test doubles for the Gemini client. No pytest magic: import and use."""

from types import SimpleNamespace


def error_body(code: int, status: str, details: list[dict] | None = None) -> dict:
    """The JSON body Gemini sends with an error response."""
    return {
        "error": {
            "code": code,
            "message": "nope",
            "status": status,
            "details": details or [],
        }
    }


def chunk(text=None, usage=None, finish=None) -> SimpleNamespace:
    """One streamed chunk. Thinking chunks have no text; the last has the finish."""
    candidates = [SimpleNamespace(finish_reason=finish)] if finish else None
    return SimpleNamespace(text=text, usage_metadata=usage, candidates=candidates)


class FailingModels:
    """generate_content always raises `error`."""

    def __init__(self, error: Exception):
        self.error = error

    def generate_content(self, **_kwargs):
        raise self.error


class StreamingModels:
    """generate_content_stream yields the chunks, then raises `error` if given."""

    def __init__(self, chunks, error: Exception | None = None):
        self.chunks = chunks
        self.error = error

    def generate_content_stream(self, **_kwargs):
        yield from self.chunks
        if self.error:
            raise self.error
