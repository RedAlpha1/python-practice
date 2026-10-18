from pydantic import BaseModel, ValidationError


class LLMParserError(Exception):
    """Raised when no valid JSON can be extracted"""


def parse_llm_json[T: BaseModel](text: str, model: type[T]) -> T:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LLMParserError("no JSON object")
    json_text = text[start : end + 1]
    try:
        return model.model_validate_json(json_text)
    except ValidationError as e:
        raise LLMParserError(f"invalid JSON for {model.__name__}: {e}") from e



if __name__ == "__main__":
    from typing import Literal

    class Summary(BaseModel):
        title: str
        sentiment: Literal["positive", "neutral", "negative"]

    cases = [
        '{"title": "x", "sentiment": "positive"}',
        'Sure! Here it is:\n```json\n{"title": "RAG", "sentiment": "neutral"}\n```\nAnything else?',
        "Sorry, I can't help with that.",
        '{"title": "x", "sentiment": "happy"}',
        '{"title": "x", "meta": {"a": 1}, "sentiment": "negative"}',
    ]
    for case in cases:
        try:
            print("OK  ", parse_llm_json(case, Summary))
        except LLMParserError as e:
            print("FAIL", e)
