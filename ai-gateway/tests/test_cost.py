from decimal import Decimal

from app.core.cost import PriceTable
from app.settings import load_pricing_config, load_providers_config

TABLE = PriceTable(
    {
        "big": {"input": 1.0, "output": 2.0},
        "free": {"input": 0, "output": 0},
    }
)


def test_cost_formula():
    # (1000 x 1.0 + 500 x 2.0) / 1,000,000 = 0.002
    assert TABLE.estimate("big", 1000, 500) == Decimal("0.002")


def test_input_and_output_use_separate_rates():
    assert TABLE.estimate("big", 1_000_000, 0) == Decimal("1")
    assert TABLE.estimate("big", 0, 1_000_000) == Decimal("2")


def test_zero_tokens_cost_zero():
    assert TABLE.estimate("big", 0, 0) == Decimal("0")


def test_free_model_costs_zero_not_none():
    assert TABLE.estimate("free", 100, 100) == Decimal("0")


def test_unknown_model_returns_none_not_zero():
    assert TABLE.estimate("kisi-ne-nahi-suna", 100, 100) is None


def test_result_is_decimal_not_float():
    assert isinstance(TABLE.estimate("big", 10, 10), Decimal)


def test_every_configured_model_has_a_price():
    """providers.yaml mein naya model jodo to pricing.yaml mein bhi rate zaroor ho."""
    priced = set(load_pricing_config()["models"])
    configured = {m for p in load_providers_config() for m in p["models"]}
    missing = configured - priced
    assert not missing, f"pricing.yaml mein rate nahi: {sorted(missing)}"