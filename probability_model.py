"""
models/probability_model.py — combines every feature into one model
probability. Per spec: output market_probability, model_probability, edge,
confidence, explanation, top contributing features.

Combination rule: each feature's weight is scaled by its own reported
confidence before normalizing. A feature that says "I don't have enough
data, 0.05 confidence" contributes almost nothing to the final number even
if its configured weight is large — this stops thin-data features from
dominating just because their static weight happens to be high.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import load_weights
from features.market_probability import MarketProbabilityFeature
from features.historical_base_rate import HistoricalBaseRateFeature
from features.external_data import ExternalDataFeature
from features.order_flow_volume import OrderFlowVolumeFeature
from features.momentum import MomentumFeature
from features.cross_market import CrossMarketFeature
from features.market_efficiency import MarketEfficiencyFeature
from models.kelly import recommend_position

FEATURE_REGISTRY = {
    "market_probability": MarketProbabilityFeature(),
    "historical_base_rate": HistoricalBaseRateFeature(),
    "external_data": ExternalDataFeature(),
    "order_flow_volume": OrderFlowVolumeFeature(),
    "momentum": MomentumFeature(),
    "cross_market": CrossMarketFeature(),
    "market_efficiency": MarketEfficiencyFeature(),
}


def run_model(context: dict, weights: dict = None):
    """
    context must carry whatever each feature needs (see each feature's
    compute() for its required keys). Returns a dict with model_prob, edge,
    confidence, explanation, and every feature's raw result for transparency.
    """
    weights = weights or load_weights()
    results = {}
    for key, feature in FEATURE_REGISTRY.items():
        try:
            results[key] = feature.compute(context)
        except Exception as e:
            results[key] = None
            results[key + "_error"] = str(e)

    weighted_sum = 0.0
    weight_total = 0.0
    for key, result in results.items():
        if result is None or key.endswith("_error"):
            continue
        w = weights.get(key, 0) * result.confidence
        weighted_sum += w * result.score
        weight_total += w

    model_prob = (weighted_sum / weight_total) if weight_total > 0 else 0.5
    market_prob = context.get("yes_price", 0.5)
    edge = model_prob - market_prob

    overall_confidence = (weight_total / sum(weights.values())) if sum(weights.values()) > 0 else 0
    overall_confidence = max(0.0, min(1.0, overall_confidence))

    top_drivers = sorted(
        [(k, r) for k, r in results.items() if r is not None and not k.endswith("_error")],
        key=lambda kv: weights.get(kv[0], 0) * kv[1].confidence,
        reverse=True,
    )[:3]

    kelly = recommend_position(model_prob, market_prob)

    return {
        "market_prob": market_prob,
        "model_prob": model_prob,
        "edge": edge,
        "confidence": overall_confidence,
        "kelly": kelly,
        "feature_results": {k: (r.__dict__ if r else None) for k, r in results.items() if not k.endswith("_error")},
        "top_drivers": [(name, r.explanation) for name, r in top_drivers],
        "weights_used": weights,
    }
