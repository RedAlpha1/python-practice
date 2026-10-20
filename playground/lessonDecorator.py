import time
from functools import wraps


def shout(text: str) -> str:
    return text.upper()


f = shout
print(f("hi"))
print(f.__name__)


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result

    return wrapper


@timed
def slow_add(a: int, b: int) -> int:
    time.sleep(0.5)
    return a + b


print(slow_add(2, 3))
print(slow_add.__name__)


@timed
def greet(name: str, excited: bool = False) -> str:
    return f"{name} {excited}"


print(greet("Prakhar", excited=True))
