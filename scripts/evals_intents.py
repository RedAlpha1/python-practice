"""Run the intent classifier over the labelled test set and report how it did.

Every result is cached as soon as it arrives, so a run that hits the daily quota
can resume tomorrow without paying again for the cases it already finished.

    uv run python scripts/evals_intents.py               # dev split (default)
    uv run python scripts/evals_intents.py --split test  # held-out score only
    uv run python scripts/evals_intents.py --prompt v1   # an older prompt (default: newest)
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai_lab.config import get_settings
from ai_lab.intents import PROMPTS, IntentResult, LabeledCase
from ai_lab.llm import (
    LLMError,
    LLMResponse,
    StructuredOutputError,
    ask_structured,
    log_usage_to,
)
from ai_lab.llm.cost import cost_usd

CASES_PATH = Path("data/evals/intent.jsonl")
CACHE_PATH = Path("data/evals/cache/intent.jsonl")
USAGE_PATH = Path("data/llm_calls.jsonl")
MODEL = get_settings().default_model
MAX_TOKENS = get_settings().max_tokens

logger = logging.getLogger(__name__)

# {"key": ..., "id": ..., "prompt_version": ..., "model": ..., "result": {...} | None,
#  "usage": {...}}
CacheEntry = dict[str, Any]


def load_cases(path: Path) -> list[LabeledCase]:
    """Every line must be valid: a typo in a label would quietly skew every score."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                cases.append(LabeledCase.model_validate_json(line))
            except ValidationError as e:
                raise SystemExit(f"{path} line {lineno}: {e}") from e
    dupes = [i for i, n in Counter(c.id for c in cases).items() if n > 1]
    if dupes:
        raise SystemExit(f"{path}: duplicate ids {dupes}")
    return cases


def cache_key(case: LabeledCase, version: str) -> str:
    """Hash of everything that can change the model's answer.

    Edit the prompt, the message, the schema, the model or max_tokens and the key
    changes, so a stale answer can never be reused by mistake.
    """
    payload = json.dumps(
        {
            "prompt": PROMPTS[version].format(text=case.text),
            "schema": IntentResult.model_json_schema(),
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
        },
        sort_keys=True,  # same dict, same string, same hash
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, CacheEntry]:
    """key -> cached entry. A half-written last line (crash mid-write) is skipped."""
    if not path.exists():
        return {}
    cache = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                cache[entry["key"]] = entry
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Skipping %s line %d: not a cache entry", path, lineno)
    return cache


def append_entry(path: Path, entry: CacheEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:  # closed (so flushed) per entry
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def classify(case: LabeledCase, version: str) -> CacheEntry:
    """One API call -> one cache entry. LLMError (quota, network) propagates.

    Unusable JSON is still the model's answer to this case, so it is cached as a
    miss rather than retried on every run.
    """
    # id, version and model are only there so a human can read the cache file.
    entry: CacheEntry = {
        "key": cache_key(case, version),
        "id": case.id,
        "prompt_version": version,
        "model": MODEL,
    }
    try:
        result, usage = ask_structured(
            PROMPTS[version].format(text=case.text),
            IntentResult,
            model=MODEL,
            max_tokens=MAX_TOKENS,
        )
    except StructuredOutputError as e:
        return {
            **entry,
            "result": None,
            "error": str(e),
            "usage": e.response.model_dump(exclude={"text"}) if e.response else None,
        }
    return {
        **entry,
        "result": result.model_dump(),
        "usage": usage.model_dump(exclude={"text"}),
    }


def run(cases: list[LabeledCase], cache: dict[str, CacheEntry], version: str) -> None:
    """Call the model for every case not yet cached; stop at the first API error."""
    todo = [c for c in cases if cache_key(c, version) not in cache]
    print(f"{len(cases) - len(todo)}/{len(cases)} cached, {len(todo)} to call")
    for case in todo:
        try:
            entry = classify(case, version)
        except LLMError as e:
            done = sum(cache_key(c, version) in cache for c in cases)
            if e.status == 429:
                print(f"Quota hit (429): {done}/{len(cases)} done, kal resume karo.")
            else:
                print(f"API error {e.status}: {e}\n{done}/{len(cases)} done.")
            return
        cache[entry["key"]] = entry
        append_entry(CACHE_PATH, entry)


def _same_city(a: str | None, b: str | None) -> bool:
    return (a or "").strip().casefold() == (b or "").strip().casefold()


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.0%})" if n else "0/0"


def report(
    cases: list[LabeledCase], cache: dict[str, CacheEntry], split: str, version: str
) -> None:
    scored = [(c, cache[k]) for c in cases if (k := cache_key(c, version)) in cache]
    n = len(scored)
    print(
        f"\n=== Intent eval [{split}], prompt {version}, {MODEL}: "
        f"{n}/{len(cases)} cases ==="
    )
    if not n:
        return

    intent_ok = city_ok = bad_output = 0
    total: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    misses: list[tuple[LabeledCase, IntentResult | None, str]] = []
    responses: list[LLMResponse] = []

    for case, entry in scored:
        if entry.get("usage"):
            responses.append(LLMResponse.model_validate({**entry["usage"], "text": ""}))
        total[case.intent] += 1
        if entry["result"] is None:
            bad_output += 1
            misses.append((case, None, entry.get("error", "")))
            continue
        got = IntentResult.model_validate(entry["result"])
        intent_hit = got.intent == case.intent
        city_hit = _same_city(got.city, case.city)
        intent_ok += intent_hit
        city_ok += city_hit
        correct[case.intent] += intent_hit
        if not (intent_hit and city_hit):
            misses.append((case, got, ""))

    print(f"Intent accuracy : {_pct(intent_ok, n)}")
    print(f"City accuracy   : {_pct(city_ok, n)}")
    print(f"Bad output      : {bad_output}")

    print("\nPer intent (correct/total):")
    for intent in sorted(total):
        print(f"  {intent:<20} {_pct(correct[intent], total[intent])}")

    # Reading held-out misses while tuning would fit the prompt to the test set.
    if misses and split != "dev":
        print(f"\nMisses: {len(misses)} (hidden: held-out cases; tune on --split dev)")
    elif misses:
        print(f"\nMisses ({len(misses)}):")
        for case, got, error in misses:
            expected = f"{case.intent}/{case.city}"
            actual = f"{got.intent}/{got.city}" if got else f"bad output: {error}"
            print(f"  #{case.id} {case.text!r}")
            print(f"      expected {expected}, got {actual}")
            if case.note:
                print(f"      note: {case.note}")

    tokens_in = sum(r.input_tokens for r in responses)
    tokens_out = sum(r.output_tokens + r.thinking_tokens for r in responses)
    costs = [cost_usd(r) for r in responses]
    cost = (
        f"${sum(c for c in costs if c is not None):.4f}"
        if all(c is not None for c in costs)
        else "unknown (model not in price table)"
    )
    print(
        f"\nUsage: {len(responses)} calls, {tokens_in} in / {tokens_out} out "
        f"tokens, cost {cost}"
    )
    if n < len(cases):
        print(f"{len(cases) - n} cases still pending: run again to resume.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the intent eval.")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--prompt", choices=list(PROMPTS), default=list(PROMPTS)[-1])
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    log_usage_to(USAGE_PATH)
    cases = load_cases(CASES_PATH)  # validate the whole file, whatever the split
    if args.split != "all":
        cases = [c for c in cases if c.split == args.split]
    cache = load_cache(CACHE_PATH)
    run(cases, cache, args.prompt)
    report(cases, cache, args.split, args.prompt)


if __name__ == "__main__":
    main()
