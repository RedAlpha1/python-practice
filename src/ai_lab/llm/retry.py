"""When and how long to wait before retrying a failed LLM call."""

import logging
import random
import time
from collections.abc import Callable
from functools import wraps

from ai_lab.llm.schemas import LLMError

logger = logging.getLogger(__name__)

# Server-side failures that are worth retrying. Others (501 Not Implemented, ...)
# will fail the same way again.
RETRYABLE_5XX = frozenset({500, 502, 503, 504})
DelayFn = Callable[[LLMError, int], float | None]


def llm_retry_delay(
    error: LLMError,
    attempt: int,  # 0 = the first retry, 1 = the second, ...
    *,
    base: float = 1.0,
    cap: float = 30.0,
    max_wait: float = 60.0,
    rand: Callable[[], float] = random.random,
) -> float | None:
    """Seconds to wait before retrying. None means: do not retry."""
    status = error.status

    # Our mistake (400/401/403/404...) or a server error that will not change:
    # resending gives the same answer.
    if status is not None and status != 429 and status not in RETRYABLE_5XX:
        return None

    if status == 429 and error.retry_after is not None:
        # The server told us exactly how long. A daily quota says "14 hours": give up.
        return error.retry_after if error.retry_after <= max_wait else None

    # Network failure (no status), 5xx, or a 429 with no hint: exponential backoff
    # with jitter, so many clients that failed together do not retry together.
    return min(cap, base * 2**attempt) * (0.5 + rand() / 2)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)  # looked up at call time, so tests can patch time.sleep


def retry_llm[**P, R](
    max_attempts: int = 3,
    *,
    sleep: Callable[[float], None] = _sleep,
    delay_for: DelayFn = llm_retry_delay,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a function that raises LLMError, when `delay_for` says it is worth it.

    `delay_for(error, attempt) -> seconds | None` defaults to llm_retry_delay. Pass
    e.g. functools.partial(llm_retry_delay, max_wait=10) for an interactive caller.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except LLMError as e:
                    delay = delay_for(e, attempt)
                    attempt += 1
                    if delay is None or attempt >= max_attempts:
                        raise
                    logger.warning(
                        "%s failed (status=%s: %s), retry %d/%d in %.1fs",
                        getattr(func, "__name__", "call"),  # partials have none
                        e.status,
                        e,
                        attempt,
                        max_attempts - 1,
                        delay,
                    )
                    sleep(delay)

        return wrapper

    return decorator
