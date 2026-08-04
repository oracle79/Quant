"""
features/base.py — every feature is its own independent, testable module.
Per spec: "each feature should return score, explanation, confidence
contribution" and "each feature should be independently testable."

A feature's `score` is its own probability estimate for "Up"/"Yes"
(0.0-1.0) — NOT a raw signal like a z-score. This keeps every feature
directly comparable and lets the model do a clean weighted average.
Confidence (0.0-1.0) says how much the feature trusts its own estimate
right now — e.g. a momentum feature with only 5 candles of data should
report low confidence, not pretend to be as sure as one with 100.
"""
from dataclasses import dataclass


@dataclass
class FeatureResult:
    name: str
    score: float          # probability estimate for "Up", 0.0-1.0
    confidence: float     # 0.0-1.0, how much to trust this estimate right now
    explanation: str      # human-readable, goes straight into reports


class Feature:
    name = "base"

    def compute(self, context: dict) -> FeatureResult:
        """context carries whatever data this feature needs — market snapshot,
        klines, external data, historical resolutions, etc. Raise clearly if
        required context is missing rather than silently guessing."""
        raise NotImplementedError
