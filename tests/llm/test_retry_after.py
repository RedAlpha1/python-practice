import pytest

from ai_lab.llm.gemini import _retry_after


def _retry_info(delay: object) -> dict:
    return {"error": {"details": [{"retryDelay": delay}]}}


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        pytest.param(_retry_info("52s"), 52.0, id="seconds"),
        pytest.param(_retry_info("52150s"), 52150.0, id="large"),
        pytest.param(_retry_info("1.5s"), 1.5, id="fractional"),
        pytest.param({"error": {"details": [{"other": 1}]}}, None, id="no-delay"),
        pytest.param(_retry_info("abc"), None, id="malformed-no-suffix"),
        pytest.param(_retry_info("abcs"), None, id="malformed-with-suffix"),
        pytest.param(_retry_info("nans"), None, id="nan"),
        pytest.param(_retry_info("-5s"), None, id="negative"),
        pytest.param(_retry_info(52), None, id="not-a-string"),
        pytest.param({"error": {"details": ["oops", None]}}, None, id="non-dict-items"),
        pytest.param(
            {"error": {"details": ["oops", {"retryDelay": "7s"}]}},
            7.0,
            id="skips-non-dict-then-finds",
        ),
        pytest.param(
            {"error": {"details": [{"retryDelay": "abcs"}, {"retryDelay": "7s"}]}},
            7.0,
            id="malformed-then-valid",
        ),
        pytest.param({"error": {"details": "nope"}}, None, id="details-not-a-list"),
        pytest.param({"error": "nope"}, None, id="error-not-a-dict"),
        pytest.param(None, None, id="details-none"),
        pytest.param([], None, id="details-not-a-dict"),
    ],
)
def test_retry_after(details, expected):
    assert _retry_after(details) == expected
