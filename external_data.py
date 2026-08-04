"""
features/external_data.py — funding rate + open interest, per spec's
crypto data list. Both are free from Binance's futures API (no key needed).

HYPOTHESIS (to be validated by backtesting, not assumed): elevated positive
funding + rising open interest often signals crowded long positioning,
which can precede short-term mean reversion. This is a real, testable
hypothesis — not a guarantee. The backtest engine decides if it earns its
15% weight or gets dialed toward zero.
"""
import math
from .base import Feature, FeatureResult


def _normcdf(x):
    t = 1 / (1 + 0.2316419 * abs(x))
    d = 0.3989423 * math.exp(-x * x / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return 1 - p if x > 0 else p


class ExternalDataFeature(Feature):
    name = "external_data"

    def compute(self, context: dict) -> FeatureResult:
        funding_history = context.get("funding_rate_history", [])
        oi_history = context.get("open_interest_history", [])

        if len(funding_history) < 5 or len(oi_history) < 5:
            return FeatureResult(
                name=self.name, score=0.5, confidence=0.1,
                explanation="Insufficient funding rate / open interest history.",
            )

        current_funding = funding_history[-1]
        mean_funding = sum(funding_history) / len(funding_history)
        var_funding = sum((f - mean_funding) ** 2 for f in funding_history) / len(funding_history)
        sd_funding = math.sqrt(var_funding) or 1e-9
        funding_z = (current_funding - mean_funding) / sd_funding

        oi_change = (oi_history[-1] - oi_history[0]) / (oi_history[0] or 1e-9)

        # crowded-long-mean-reversion hypothesis: high positive funding z-score
        # + rising OI -> lean bearish (mean reversion); the reverse for shorts
        lean = -funding_z * 0.4 if oi_change > 0 else 0.0
        score = _normcdf(lean)

        confidence = min(1.0, len(funding_history) / 50)
        return FeatureResult(
            name=self.name, score=score, confidence=confidence,
            explanation=f"Funding rate z-score {funding_z:.2f}, OI change {oi_change*100:+.1f}% "
                        f"-> {'bearish' if lean<0 else 'bullish' if lean>0 else 'neutral'} lean "
                        f"(crowded-positioning hypothesis, unvalidated).",
        )
