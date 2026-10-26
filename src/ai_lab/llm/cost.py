"""What a call would cost at paid-tier prices. Unknown models have no price: never guess."""

from pydantic import BaseModel

from ai_lab.llm.schemas import LLMResponse


class Price(BaseModel):
    input_per_m: float  # USD per 1M input tokens
    output_per_m: (
        float  # USD per 1M output tokens (thinking is billed at this rate too)
    )


# Paid-tier, standard (not batch, not cached) text rates, USD per 1M tokens.
# Source: https://ai.google.dev/gemini-api/docs/pricing  (checked 2026-10-26).
# Prices change: re-check the page before trusting a number. A model that is not
# listed here (e.g. gemini-3.5-flash, which the page did not show) costs None.
PRICES: dict[str, Price] = {
    "gemini-3.5-flash-lite": Price(input_per_m=0.30, output_per_m=2.50),
    # Introductory price until 2026-12-31; from 2027-01-01 it is 1.50 in / 7.50 out.
    "gemini-3.6-flash": Price(input_per_m=0.75, output_per_m=3.75),
    "gemini-2.5-flash-lite": Price(input_per_m=0.10, output_per_m=0.40),
    "gemini-2.5-flash": Price(input_per_m=0.30, output_per_m=2.50),
}


def cost_usd(r: LLMResponse) -> float | None:
    """USD for this call at paid-tier rates, or None if we don't know the model's price.

    On the free tier the real bill is 0; this is "what it would cost if paid".
    """
    price = PRICES.get(r.model)
    if price is None:
        return None
    billed_output = r.output_tokens + r.thinking_tokens  # thinking bills as output
    return (
        r.input_tokens * price.input_per_m + billed_output * price.output_per_m
    ) / 1e6
