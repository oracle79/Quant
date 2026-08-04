"""
features/cross_market.py — if a same-window market exists for a correlated
asset (e.g. ETH up/down alongside BTC up/down), its implied probability is
a weak independent signal, since crypto often moves as a block. Weighted
lowest of the "real" features (5%) precisely because this is the most
speculative hypothesis in the set.
"""
from .base import Feature, FeatureResult


class CrossMarketFeature(Feature):
    name = "cross_market"

    def compute(self, context: dict) -> FeatureResult:
        correlated_yes_price = context.get("correlated_asset_yes_price")
        correlated_asset = context.get("correlated_asset_name", "correlated asset")

        if correlated_yes_price is None:
            return FeatureResult(name=self.name, score=0.5, confidence=0.0,
                                  explanation="No correlated same-window market available.")

        # pull gently toward the correlated market's price, don't fully copy it
        score = 0.5 + (correlated_yes_price - 0.5) * 0.5
        return FeatureResult(
            name=self.name, score=score, confidence=0.25,
            explanation=f"{correlated_asset}'s same-window market is pricing "
                        f"{correlated_yes_price*100:.1f}% Up.",
        )
