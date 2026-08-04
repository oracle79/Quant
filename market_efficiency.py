"""
features/market_efficiency.py — thin, wide-spread markets are less reliably
priced than deep, tight ones. This feature doesn't add directional opinion;
it nudges score gently toward 0.5 when the market looks illiquid/inefficient,
reflecting the idea that an inefficient market's own price deserves less
trust — separate from (and much smaller than) the main market_probability
feature's own confidence scaling.
"""
from .base import Feature, FeatureResult


class MarketEfficiencyFeature(Feature):
    name = "market_efficiency"

    def compute(self, context: dict) -> FeatureResult:
        yes_price = context["yes_price"]
        spread = context.get("spread")
        liquidity = context.get("liquidity", 0)

        if spread is None or liquidity < 100:
            # can't assess efficiency reliably -> mild pull to neutral, low confidence
            score = 0.5 + (yes_price - 0.5) * 0.5
            return FeatureResult(name=self.name, score=score, confidence=0.15,
                                  explanation="Insufficient data to assess market efficiency; "
                                              "discounting toward neutral as a precaution.")

        # wide spread -> less efficient -> pull toward neutral more
        inefficiency = min(1.0, spread / 0.05)  # 5c+ spread treated as very wide
        score = yes_price * (1 - inefficiency * 0.3) + 0.5 * (inefficiency * 0.3)
        return FeatureResult(
            name=self.name, score=score, confidence=0.3,
            explanation=f"Spread {spread*100:.1f}c, liquidity ${liquidity:,.0f} -> "
                        f"{'discounting price reliability' if inefficiency>0.3 else 'market looks efficient'}.",
        )
