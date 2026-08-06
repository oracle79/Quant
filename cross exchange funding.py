"""
features/cross_exchange_funding.py — compares Binance's funding rate to
Bybit's for the same window. A single exchange's funding rate reflects
that exchange's own trader positioning; when two major exchanges diverge
meaningfully, it can reflect real information asymmetry or an isolated
crowded-positioning pocket on one venue — a genuinely different hypothesis
from external_data's single-exchange view, not a duplicate of it.

HYPOTHESIS (unvalidated until backtested): when Binance's funding runs
much hotter (more positive) than Bybit's, Binance-side longs are unusually
crowded relative to the wider market — lean bearish, and vice versa.
"""
import math
from .base import Feature, FeatureResult


def _normcdf(x):
    t = 1 / (1 + 0.2316419 * abs(x))
    d = 0.3989423 * math.exp(-x * x / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return 1 - p if x > 0 else p


class CrossExchangeFundingFeature(Feature):
    name = "cross_exchange_funding"

    def compute(self, context: dict) -> FeatureResult:
        binance_hist = context.get("binance_funding_history", [])
        bybit_hist = context.get("bybit_funding_history", [])

        if len(binance_hist) < 3 or len(bybit_hist) < 3:
            return FeatureResult(name=self.name, score=0.5, confidence=0.05,
                                  explanation="Insufficient funding history on one or both exchanges.")

        current_divergence = binance_hist[-1] - bybit_hist[-1]
        history_len = min(len(binance_hist), len(bybit_hist))
        divergences = [binance_hist[-i] - bybit_hist[-i] for i in range(1, history_len + 1)]
        mean_div = sum(divergences) / len(divergences)
        var_div = sum((d - mean_div) ** 2 for d in divergences) / len(divergences)
        sd_div = math.sqrt(var_div) or 1e-9
        z = (current_divergence - mean_div) / sd_div
        z = max(-4.0, min(4.0, z))

        # crowded-Binance-longs hypothesis: hot positive divergence -> lean bearish
        lean = -z * 0.35
        score = _normcdf(lean)

        confidence = min(1.0, history_len / 30)
        return FeatureResult(
            name=self.name, score=score, confidence=confidence,
            explanation=f"Binance-Bybit funding divergence z-score {z:.2f} "
                        f"-> {'bearish' if lean<0 else 'bullish' if lean>0 else 'neutral'} lean "
                        f"(cross-exchange positioning hypothesis, unvalidated).",
        )
