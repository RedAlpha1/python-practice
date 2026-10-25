import httpx
import pytest
from google.genai import types

from ai_lab.llm.gemini import generate_turn
from ai_lab.llm.schemas import LLMError, ToolCall, ToolSpec, Turn, UserText
from tests.llm.fakes import RecordingModels, fake_response

TOOLS = [
    ToolSpec(
        name="add",
        description="Add two integers",
        parameter={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
]
TURNS: list[Turn] = [UserText(text="17 + 25?")]


def test_tool_call_becomes_assistant_turn(install_models):
    call = types.FunctionCall(id="c1", name="add", args={"a": 17, "b": 25})
    resp = fake_response([types.Part(function_call=call)])
    models = RecordingModels(resp)
    install_models(models)

    turn, response = generate_turn(TURNS, TOOLS)

    assert turn.tool_calls == [ToolCall(id="c1", name="add", args={"a": 17, "b": 25})]
    assert turn.text == ""
    assert resp.candidates is not None
    assert turn.provider_raw is resp.candidates[0].content  # kept for the next round
    assert (response.input_tokens, response.output_tokens) == (11, 4)
    assert response.thinking_tokens == 3
    assert response.finish_reason == "stop"


def test_plain_text_answer_has_no_tool_calls(install_models):
    install_models(RecordingModels(fake_response([types.Part(text="It is 42.")])))

    turn, response = generate_turn(TURNS, TOOLS)

    assert turn.text == "It is 42."
    assert turn.tool_calls == []
    assert response.text == "It is 42."


def test_thought_parts_are_not_part_of_the_text(install_models):
    parts = [
        types.Part(text="secret reasoning", thought=True),
        types.Part(text="Answer."),
    ]
    install_models(RecordingModels(fake_response(parts)))

    turn, _ = generate_turn(TURNS, TOOLS)

    assert turn.text == "Answer."


def test_tools_are_declared_and_auto_calling_is_off(install_models):
    models = RecordingModels(fake_response([types.Part(text="ok")]))
    install_models(models)

    generate_turn(TURNS, TOOLS, system="be brief")

    config = models.kwargs["config"]
    [tool] = config.tools
    [decl] = tool.function_declarations
    assert decl.name == "add"
    assert decl.description == "Add two integers"
    assert decl.parameters_json_schema == TOOLS[0].parameter
    assert config.automatic_function_calling.disable is True
    assert config.system_instruction == "be brief"
    assert models.kwargs["contents"][0].role == "user"


def test_no_tools_means_no_tool_config(install_models):
    models = RecordingModels(fake_response([types.Part(text="ok")]))
    install_models(models)

    generate_turn(TURNS, [])

    assert models.kwargs["config"].tools is None


def test_max_tokens_finish_is_normalized(install_models):
    resp = fake_response(
        [types.Part(text="cut of")], finish=types.FinishReason.MAX_TOKENS
    )
    install_models(RecordingModels(resp))

    _, response = generate_turn(TURNS, TOOLS)

    assert response.finish_reason == "max_tokens"


def test_network_error_becomes_llm_error(install_models):
    install_models(RecordingModels(error=httpx.ConnectError("dns")))

    with pytest.raises(LLMError) as info:
        generate_turn(TURNS, TOOLS)

    assert info.value.status is None
