import pytest

from ai_lab.text import chunk_text


@pytest.mark.parametrize(
    ("text", "size", "overlap", "expected"),
    [
        pytest.param("abcdefghij", 4, 1, ["abcd", "defg", "ghij"], id="overlap"),
        pytest.param("abcdefgh", 4, 0, ["abcd", "efgh"], id="no-overlap"),
        pytest.param("abc", 4, 1, ["abc"], id="shorter-than-size"),
        pytest.param("", 4, 1, [], id="empty-text"),
    ],
)
def test_chunk_text(text, size, overlap, expected):
    assert list(chunk_text(text, size=size, overlap=overlap)) == expected


def test_bad_overlap_raises():
    # list() is needed: chunk_text is a generator, so the guard only runs on first next()
    with pytest.raises(ValueError, match="overlap"):
        list(chunk_text("abc", size=2, overlap=2))


def test_bad_size_raises():
    with pytest.raises(ValueError, match="size"):
        list(chunk_text("abc", size=0))
