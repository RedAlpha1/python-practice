import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from google.genai import types

from ai_lab.config import get_settings
from ai_lab.llm.gemini import _client  # playground hai, isliye private import chalega


# 1. Asli Python functions
def get_time(timezone: str) -> str:
    return datetime.now(ZoneInfo(timezone)).strftime("%H:%M")


def add(a: int, b: int) -> int:
    return a + b


# 2. Model ko functions ke baare mein batao
get_time_decl = types.FunctionDeclaration(
    name="get_time",
    description="Get the current time in a timezone",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "e.g. Asia/Tokyo"}
        },
        "required": ["timezone"],
    },
)

add_decl = types.FunctionDeclaration(
    name="add",
    description="Add two integers and return their sum",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "first number"},
            "b": {"type": "integer", "description": "second number"},
        },
        "required": ["a", "b"],
    },
)

config = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[get_time_decl, add_decl])],
)
TOOLS = {"get_time": get_time, "add": add}
MAX_ROUNDS = 5


def run_tool(call: types.FunctionCall) -> dict:
    """Tool chalao. Gemini ka convention: success pe "output", failure pe "error"."""
    name = call.name or ""
    if name not in TOOLS:
        return {"error": f"unknown tool '{name}'. Available: {list(TOOLS)}"}
    try:
        return {"output": TOOLS[name](**(call.args or {}))}
    except Exception as e:  # noqa: BLE001 - error bhi model ko bhejna hai, crash nahi
        return {"error": f"{type(e).__name__}: {e}"}


def main(question: str) -> None:
    model = get_settings().default_model
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    # 3-6. Loop: model tool maange -> hum chalayein -> result wapas, jab tak model
    # seedha jawab na de. Max MAX_ROUNDS rounds, warna ruk jao.
    for step in range(1, MAX_ROUNDS + 1):
        resp = _client().models.generate_content(
            model=model, contents=contents, config=config
        )
        if not resp.function_calls:
            print(f"Final jawab (round {step}):", resp.text)
            break

        # Model ek hi turn mein kai tools maang sakta hai, isliye sabko chalao
        response_parts = []
        for call in resp.function_calls:
            response = run_tool(call)
            print(f"[round {step}] {call.name}({call.args}) -> {response}")
            response_parts.append(
                types.Part.from_function_response(
                    name=call.name or "", response=response
                )
            )

        # Conversation mein jodo: model ka turn (jaisa aaya waisa) + hamare results
        assert resp.candidates and resp.candidates[0].content is not None
        contents.append(resp.candidates[0].content)
        contents.append(types.Content(role="user", parts=response_parts))
    else:
        print(f"{MAX_ROUNDS} rounds ke baad bhi model tools maang raha hai, ruk gaye")


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "What time is it in Tokyo, and what is 17 plus 25?"
    )
