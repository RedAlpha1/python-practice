import logging
import time
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logger.info("%s took %.3fs", func.__name__, time.perf_counter() - start)

    return wrapper


def retry(
    times: int = 3,
    backoff: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    if times < 1:
        raise ValueError("times must be >= 1")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == times:
                        raise
                    delay = backoff * 2 ** (attempt - 1)
                    logger.warning(
                        "%s failed (%s), attempt %d/%d, retrying in %.1fs",
                        func.__name__,
                        e,
                        attempt,
                        times,
                        delay,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        logger.info(
            "%s took %.3fs", self.label, self.elapsed
        )  # builds it only if the log is shown
        return False


@contextmanager
def timer(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s took %.3fs", label, elapsed)


if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def make_flaky(fail_rate: float, error: type[Exception], exceptions):
        @timed
        @retry(times=4, backoff=0.1, exceptions=exceptions)
        def flaky() -> str:
            if random.random() < fail_rate:
                raise error("network blip" if error is ConnectionError else "bug")
            return "ok"

        return flaky

    scenarios = [
        ("70% failure", 0.7, ConnectionError, (ConnectionError,)),
        ("100% failure", 1.0, ConnectionError, (ConnectionError,)),
        ("bug not retried", 1.0, ValueError, (ConnectionError,)),
    ]

    for cm in (Timer("class version"), timer("generator version")):
        try:
            with cm:
                time.sleep(0.1)
                raise ValueError("boom")
        except ValueError:
            logger.info("caught it")

    for name, rate, error, exc in scenarios:
        random.seed(1)
        logger.info("--- %s", name)
        try:
            logger.info("result: %s", make_flaky(rate, error, exc)())
        except (ConnectionError, ValueError) as e:
            logger.info("raised %s: %s", type(e).__name__, e)
