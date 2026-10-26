from functools import partial

import pytest

from ai_lab.llm.retry import llm_retry_delay, retry_llm
from ai_lab.llm.schemas import LLMError


def error(status: int | None, retry_after: float | None = None) -> LLMError:
    return LLMError("boom", status=status, retry_after=retry_after)


# rand=1.0 makes the jitter factor exactly 1.0, so the answers are fixed numbers.
@pytest.mark.parametrize(
    ("err", "attempt", "expected"),
    [
        pytest.param(error(None), 0, 1.0, id="network-first-retry"),
        pytest.param(error(None), 3, 8.0, id="network-doubles"),
        pytest.param(error(503), 2, 4.0, id="503-attempt-2"),
        pytest.param(error(500), 0, 1.0, id="500"),
        pytest.param(error(502), 1, 2.0, id="502"),
        pytest.param(error(504), 1, 2.0, id="504"),
        pytest.param(error(503), 10, 30.0, id="capped-at-30"),
        pytest.param(error(429, 52), 0, 52.0, id="429-wait-exactly-retry-after"),
        pytest.param(error(429, 52), 5, 52.0, id="429-ignores-attempt"),
        pytest.param(error(429, 60), 0, 60.0, id="429-at-the-limit-still-retries"),
        pytest.param(error(429, 52150), 0, None, id="429-daily-quota-gives-up"),
        pytest.param(error(429), 1, 2.0, id="429-no-hint-uses-backoff"),
        pytest.param(error(400), 0, None, id="400"),
        pytest.param(error(401), 0, None, id="401"),
        pytest.param(error(403), 0, None, id="403"),
        pytest.param(error(404), 0, None, id="404"),
        pytest.param(error(501), 0, None, id="501-not-implemented-never-retried"),
        pytest.param(error(505), 0, None, id="505"),
        pytest.param(error(408), 0, None, id="408"),
    ],
)
def test_llm_retry_delay(err, attempt, expected):
    assert llm_retry_delay(err, attempt, rand=lambda: 1.0) == expected


def test_jitter_scales_the_backoff_between_half_and_full():
    err = error(503)
    assert llm_retry_delay(err, 2, rand=lambda: 0.0) == 2.0  # 4s * 0.5
    assert llm_retry_delay(err, 2, rand=lambda: 1.0) == 4.0  # 4s * 1.0


def test_custom_base_cap_and_max_wait():
    assert llm_retry_delay(error(None), 1, base=0.1, rand=lambda: 1.0) == 0.2
    assert llm_retry_delay(error(503), 9, cap=5, rand=lambda: 1.0) == 5.0
    assert llm_retry_delay(error(429, 20), 0, max_wait=10) is None


# ── the decorator ─────────────────────────────────────────────────────
class Flaky:
    """Raises the queued errors one per call, then returns "ok"."""

    def __init__(self, *errors: LLMError):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self, who: str = "x") -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return f"ok {who}"


def test_retries_after_a_503_then_succeeds():
    slept: list[float] = []
    flaky = Flaky(error(503))

    result = retry_llm(sleep=slept.append)(flaky)("bob")

    assert result == "ok bob"  # arguments and the result pass through
    assert flaky.calls == 2
    assert len(slept) == 1
    assert 0.5 <= slept[0] <= 1.0  # first retry: 1s with jitter


def test_400_raises_immediately_and_never_sleeps():
    slept: list[float] = []
    flaky = Flaky(error(400))

    with pytest.raises(LLMError) as info:
        retry_llm(sleep=slept.append)(flaky)()

    assert info.value.status == 400
    assert flaky.calls == 1
    assert slept == []


def test_gives_up_after_max_attempts_and_raises_the_last_error():
    slept: list[float] = []
    flaky = Flaky(error(503), error(503), error(503))

    with pytest.raises(LLMError):
        retry_llm(max_attempts=3, sleep=slept.append)(flaky)()

    assert flaky.calls == 3
    assert len(slept) == 2  # no sleep after the last failure


def test_429_waits_exactly_retry_after():
    slept: list[float] = []
    flaky = Flaky(error(429, 7))

    retry_llm(sleep=slept.append)(flaky)()

    assert slept == [7.0]


def test_daily_quota_429_fails_at_once():
    slept: list[float] = []
    flaky = Flaky(error(429, 52150))

    with pytest.raises(LLMError):
        retry_llm(sleep=slept.append)(flaky)()

    assert flaky.calls == 1
    assert slept == []


def test_other_exceptions_are_not_retried():
    def broken() -> None:
        raise KeyError("bug")

    with pytest.raises(KeyError):
        retry_llm(sleep=lambda _s: None)(broken)()


def test_wraps_keeps_the_function_name():
    @retry_llm()
    def ask_something() -> str:
        return "x"

    assert ask_something.__name__ == "ask_something"


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError, match="max_attempts"):
        retry_llm(max_attempts=0)


def test_custom_delay_function_is_used():
    slept: list[float] = []
    flaky = Flaky(error(503))

    retry_llm(sleep=slept.append, delay_for=lambda _err, _attempt: 0.25)(flaky)()

    assert slept == [0.25]


def test_custom_delay_function_can_refuse_to_retry():
    slept: list[float] = []
    flaky = Flaky(error(503))

    with pytest.raises(LLMError):
        retry_llm(sleep=slept.append, delay_for=lambda _err, _attempt: None)(flaky)()

    assert flaky.calls == 1
    assert slept == []


def test_interactive_caller_gives_up_on_a_long_429_wait():
    slept: list[float] = []
    flaky = Flaky(error(429, 30))
    short_wait = partial(llm_retry_delay, max_wait=10)  # a user will not wait 30s

    with pytest.raises(LLMError):
        retry_llm(sleep=slept.append, delay_for=short_wait)(flaky)()

    assert flaky.calls == 1
    assert slept == []
