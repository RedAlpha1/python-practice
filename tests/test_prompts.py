from ai_lab.models import Conversation, Message
from ai_lab.prompts import build_summary_prompt


def test_summary_prompt_shows_last_six_of_ten():
    msgs = [
        Message(role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
        for i in range(10)
    ]
    conv = Conversation(
        id="a", model="gpt", started_at="2026-10-01T10:00:00", messages=msgs
    )

    prompt = build_summary_prompt(conv)

    assert "Messages shown: 6 of 10" in prompt
