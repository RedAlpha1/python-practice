import random
from datetime import datetime, timedelta
from pathlib import Path

from ai_lab.models import Conversation, Message, save_conversations

random.seed(42)
BASE_DATE = datetime(2026, 10, 17, 12, 0, 0)

MODELS = ["claude", "gpt", "gemini"]
QUESTIONS = [
    "How do I reverse a list in Python?",
    "What is the difference between a list and a tuple?",
    "Explain RAG in simple terms.",
    "How do I read a JSON file in Python?",
    "What does async/await do?",
    "How do embeddings work?",
    "What is a dataclass?",
    "How can I speed up my API calls?",
]

ANSWERS = [
    "Use my_list[::-1] or my_list.reverse().",
    "Lists are mutable, tuples are immutable.",
    "RAG retrieves relevant documents and adds them to the prompt before the model answers.",
    "Open the file and call json.load(f).",
    "It lets your program wait for I/O without blocking other tasks.",
    "They turn text into vectors so similar meanings end up close together.",
    "A class decorator that generates __init__, __repr__ and __eq__ for you.",
    "Run them concurrently with asyncio.gather and cap concurrency with a Semaphore.",
]


def make_conversation(i: int) -> Conversation:
    count = random.randint(2, 8)
    messages = [
        Message(
            role="user" if n % 2 == 0 else "assistant",
            content=random.choice(QUESTIONS) if n % 2 == 0 else random.choice(ANSWERS),
        )
        for n in range(count)
    ]
    started_at = (BASE_DATE - timedelta(days=random.randint(0, 14))).isoformat()
    return Conversation(
        id=f"c{i:03d}",
        model=random.choice(MODELS),
        started_at=started_at,
        messages=messages,
    )


if __name__ == "__main__":
    chats = [make_conversation(i) for i in range(30)]
    save_conversations(chats, Path("data/chats.jsonl"))
    print(f"Wrote {len(chats)} chats")
