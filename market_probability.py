"""
features/market_probability.py — the market's own current implied
probability. Weighted highest (45%) in the starting hypothesis because
Polymarket prices already aggregate a lot of information; other features
exist to find where that aggregate might still be wrong, not to replace it.
"""
from .base import Feature, FeatureResult


class MarketProbabilityFeature(Feature):
    name = "market_probability"

    def compute(self, context: dict) -> FeatureResult:
        yes_price = context["yes_price"]
        liquidity = context.get("liquidity", 0)
        # confidence scales with liquidity — a $50 market's price means much
        # less than a $50,000 market's price
        confidence = min(1.0, liquidity / 5000) if liquidity else 0.3
        return FeatureResult(
            name=self.name,
            score=yes_price,
            confidence=max(0.2, confidence),
            explanation=f"Market currently prices this at {yes_price*100:.1f}% "
                        f"(liquidity ${liquidity:,.0f})",
        )
