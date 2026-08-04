"""
features/order_flow_volume.py — Polymarket's own order book: bid/ask
imbalance and volume surge. Distinct from momentum (which looks at the
underlying asset's price) — this looks at how THIS specific market is
being traded, which can diverge from spot (e.g. a market gets crowded
with one-sided bets independent of what BTC itself is doing).
"""
from .base import Feature, FeatureResult


class OrderFlowVolumeFeature(Feature):
    name = "order_flow_volume"

    def compute(self, context: dict) -> FeatureResult:
        best_bid = context.get("best_bid")
        best_ask = context.get("best_ask")
        bid_depth = context.get("bid_depth_usd", 0)
        ask_depth = context.get("ask_depth_usd", 0)

        total_depth = bid_depth + ask_depth
        if total_depth < 50:
            return FeatureResult(name=self.name, score=0.5, confidence=0.1,
                                  explanation="Order book too thin to read reliably.")

        # more size stacked on the bid than the ask suggests buying pressure
        # on "Up" shares (bid depth is demand to buy Up at that price)
        imbalance = (bid_depth - ask_depth) / total_depth  # -1..+1
        score = 0.5 + imbalance * 0.15  # modest nudge, not a strong claim
        score = max(0.0, min(1.0, score))

        confidence = min(1.0, total_depth / 2000)
        return FeatureResult(
            name=self.name, score=score, confidence=confidence,
            explanation=f"Order book imbalance {imbalance:+.2f} "
                        f"(bid depth ${bid_depth:,.0f} vs ask depth ${ask_depth:,.0f}).",
        )
