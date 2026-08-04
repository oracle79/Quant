"""
features/momentum.py — recent price momentum normalized by realized
volatility. This is the same core logic already backtested today (result:
no standalone edge at n=500 — see backtest_runs history). Kept in the
modular system specifically SO the backtest engine can keep re-checking
it as more data accumulates, rather than deciding once and never revisiting.
"""
import math
from .base import Feature, FeatureResult


def _normcdf(x):
    t = 1 / (1 + 0.2316419 * abs(x))
    d = 0.3989423 * math.exp(-x * x / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return 1 - p if x > 0 else p


class MomentumFeature(Feature):
    name = "momentum"

    def compute(self, context: dict) -> FeatureResult:
        closes = context["closes"]
        if len(closes) < 10:
            return FeatureResult(name=self.name, score=0.5, confidence=0.05,
                                  explanation="Insufficient price history.")

        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        mean_ret = sum(rets) / len(rets)
        variance = sum((r - mean_ret) ** 2 for r in rets) / len(rets)
        vol = math.sqrt(variance) or 1e-6

        cum_ret = math.log(closes[-1] / closes[0])
        z = cum_ret / (vol * math.sqrt(len(closes)))
        z = max(-6.0, min(6.0, z))  # guard against a bad tick or near-zero vol blowing this up
        score = _normcdf(z)

        return FeatureResult(
            name=self.name, score=score, confidence=0.5,
            explanation=f"Momentum z-score {z:.2f} over {len(closes)} candles.",
        )
