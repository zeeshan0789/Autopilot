import logging
from decimal import Decimal

from app.settings import load_pricing_config

logger = logging.getLogger("autopilot.cost")

ONE_MILLION = Decimal(1_000_000)


class PriceTable:
    def __init__(self, models: dict):
        # float nahi, Decimal: paison ke hisab mein chhoti galtiyan jama na hon
        self._prices = {
            name: (Decimal(str(entry["input"])), Decimal(str(entry["output"])))
            for name, entry in models.items()
        }

    def estimate(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> Decimal | None:
        """Estimated cost (USD). Model ka rate na mile to None (0 nahi)."""
        prices = self._prices.get(model)
        if prices is None:
            logger.warning("pricing.yaml mein '%s' ka rate nahi hai, cost None", model)
            return None
        input_price, output_price = prices
        return (
            Decimal(prompt_tokens) * input_price
            + Decimal(completion_tokens) * output_price
        ) / ONE_MILLION


price_table = PriceTable(load_pricing_config()["models"])


def estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> Decimal | None:
    return price_table.estimate(model, prompt_tokens, completion_tokens)