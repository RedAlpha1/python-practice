import pytest

from ai_lab.llm import cost
from ai_lab.llm.cost import PRICES as REAL_PRICES
from ai_lab.llm.cost import Price, cost_usd
from ai_lab.llm.schemas import LLMResponse


@pytest.fixture(autouse=True)
def fixed_prices(monkeypatch):
    """Test with our own prices, so a price change on the real page cannot break us."""
    monkeypatch.setattr(
        cost, "PRICES", {"priced": Price(input_per_m=2.0, output_per_m=10.0)}
    )


def response(
    model: str = "priced",
    input_tokens: int = 0,
    output_tokens: int = 0,
    thinking_tokens: int = 0,
) -> LLMResponse:
    return LLMResponse(
        text="",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        finish_reason="stop",
        model=model,
        latency_ms=0,
        first_token_ms=None,
    )


def test_one_million_input_tokens_cost_the_input_rate():
    assert cost_usd(response(input_tokens=1_000_000)) == pytest.approx(2.0)


def test_one_million_output_tokens_cost_the_output_rate():
    assert cost_usd(response(output_tokens=1_000_000)) == pytest.approx(10.0)


def test_thinking_tokens_are_billed_at_the_output_rate():
    # 587 thinking tokens for a 55-token answer: the thinking is most of the bill.
    r = response(input_tokens=8, output_tokens=55, thinking_tokens=587)
    expected = (8 * 2.0 + (55 + 587) * 10.0) / 1_000_000
    assert cost_usd(r) == pytest.approx(expected)

    without_thinking = cost_usd(response(input_tokens=8, output_tokens=55))
    assert without_thinking is not None
    assert expected > 5 * without_thinking  # thinking is most of the bill


def test_unknown_model_has_no_price_never_a_guess():
    assert cost_usd(response(model="mystery", input_tokens=1_000_000)) is None


def test_zero_tokens_cost_zero_for_a_known_model():
    assert cost_usd(response()) == 0.0


def test_the_real_price_table_is_sane():
    # REAL_PRICES is the module's own dict, bound before the fixture patched the name.
    assert REAL_PRICES, "no prices filled in"
    for name, price in REAL_PRICES.items():
        assert price.input_per_m > 0, name
        assert price.output_per_m >= price.input_per_m, name  # output costs more
