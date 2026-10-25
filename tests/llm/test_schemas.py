import pytest
from pydantic import ValidationError

from ai_lab.llm.schemas import ToolResult


def test_tool_result_rejects_output_and_error_together():
    with pytest.raises(ValidationError, match="both output and error"):
        ToolResult(name="add", output=5, error="failed")


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"output": 5}, id="success"),
        pytest.param({"error": "boom"}, id="failure"),
        pytest.param({"error": ""}, id="empty-error-is-still-an-error"),
        pytest.param({"output": 0}, id="falsy-output-is-still-output"),
        pytest.param({}, id="tool-returned-nothing"),
    ],
)
def test_tool_result_allows_one_or_neither(fields):
    ToolResult(name="add", **fields)
