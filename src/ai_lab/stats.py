from ai_lab.models import Conversation


def total_message_per_model(convs: list[Conversation]) -> dict[str, str]:
    unique_models = {c.model for c in convs}
    return {
        model: sum(len(c.messages) for c in convs if c.model == model)
        for model in sorted(unique_models)
    }


def avg_turns(convs: list[Conversation]) -> float:
    if not convs:
        return 0.0
    total_messages = sum(len(c.messages) for c in convs)
    return total_messages / len(convs)
