from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass()
class Conversation:
    model: str
    messages: list[Message]
    id: str

    def user_turn(self) -> int:
        return sum(1 for msg in self.messages if msg.role == "user")

    def last_user_message(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    def transcript(self) -> str:
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self.messages)


# Build 2 conversations and print all three methods for each.

conversation1 = Conversation(
    id="c1",
    model="gpt-4o",
    messages=[Message(role="user", content="Hello, how are you?")],
)
conversation2 = Conversation(
    id="c2",
    model="gpt-4o",
    messages=[
        Message(role="user", content="Hello, how are you?"),
        Message(role="assistant", content="I'm fine, thank you!"),
    ],
)

print(conversation1.user_turn())
print(conversation1.last_user_message())
print(conversation1.transcript())

print(conversation2.user_turn())
print(conversation2.last_user_message())
print(conversation2.transcript())

# Convert this into a Conversation. Hint: build the Message objects with a comprehension first.

raw_conversation = {
    "id": "c1",
    "model": "claude",
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ],
}

messages = [Message(**m) for m in raw_conversation["messages"]]
conversation3 = Conversation(**{**raw_conversation, "messages": messages})

# conversation3 = Conversation(
#     id=raw_conversation["id"],
#     model=raw_conversation["model"],
#     messages=[
#         Message(role=msg["role"], content=msg["content"])
#         for msg in raw_conversation["messages"]
#     ],
# )
print(conversation3.user_turn())
print(conversation3.last_user_message())
print(conversation3.transcript())

# Try msg.content = "changed" on a frozen Message, see the error it raises, and tell me what it says.
conversation3.messages[0].content = "changed"
