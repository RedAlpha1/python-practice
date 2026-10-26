import sys
from pathlib import Path

from ai_lab.llm.report import format_table, load_usage

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/llm_calls.jsonl")

if not path.exists():
    raise SystemExit(
        f"No usage file at {path}. Run a script that calls log_usage_to first."
    )

rows = load_usage(path)
if not rows:
    raise SystemExit(f"{path} has no valid records.")

print(format_table(rows))
