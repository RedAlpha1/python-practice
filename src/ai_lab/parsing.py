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
