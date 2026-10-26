import sys
from pathlib import Path

from ai_lab.llm import ask, log_usage_to

log_usage_to(Path("data/llm_calls.jsonl"))

question = "Explain RAG in 2 lines"
max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else None
system = sys.argv[2] if len(sys.argv) > 2 else None

result = ask(question, system=system, max_tokens=max_tokens)

print("answer:       ", result.text)
print("input tokens: ", result.input_tokens)
print("output tokens:", result.output_tokens)
print("think tokens: ", result.thinking_tokens)
print("finish reason:", result.finish_reason)
