import json
from pathlib import Path
from ai_lab.models import Conversation

path = Path("data/chats.jsonl")


def countdown(n : int):
    print("START")
    while n > 0:
        print(f" about to yield {n}")
        yield n
        print(f"  resumed after {n}")
        n -= 1
    print("done")    

gen = countdown(3)
print("created generator")


print(next(gen))
print(next(gen))
print("--- now a for loop for the rest")
for x in gen:
    print(x)


def read_lines(path : Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

convs = (Conversation.model_validate_json(line) for line in read_lines(path))
print(next(convs).id) 


gen2 = read_lines(path)
print(len(list(gen2)))
print(len(list(gen2)))


