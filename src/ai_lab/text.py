from collections.abc import Iterator


def chunk_text(text: str, size: int, overlap: int = 0) -> Iterator[str]:
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, {size}), got {overlap}")

    step = size - overlap
    start = 0
    while start < len(text):
        yield text[start : start + size]
        if start + size >= len(text):
            break
        start += step


if __name__ == "__main__":
    print(list(chunk_text("abcdefghij", 4, 1)))
