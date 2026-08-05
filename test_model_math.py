"""
tests/test_model_math.py — plain-assertion tests, no pytest needed.
Run with: python3 tests/test_model_math.py
Covers the math that must never silently break: Kelly formula, Brier
score, log loss, and bootstrap CI correctly separating noise from signal.
"""
import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kelly import kelly_fraction, recommend_position
from backtesting.metrics import brier_score, log_loss, max_drawdown
from backtesting.monte_carlo import bootstrap_accuracy_ci


def test_kelly_fraction_no_edge_at_fair_price():
    # if model agrees exactly with market price, there's no edge -> Kelly should be ~0
    f = kelly_fraction(0.5, 0.5)
    assert abs(f) < 1e-9, f"expected ~0 Kelly at fair price, got {f}"
    print("PASS: kelly_fraction returns 0 at fair price")


def test_kelly_fraction_positive_edge():
    # model says 60% but market prices 50% -> real edge -> positive Kelly
    f = kelly_fraction(0.60, 0.50)
    assert f > 0, f"expected positive Kelly with real edge, got {f}"
    print("PASS: kelly_fraction is positive with a real edge")


def test_kelly_fraction_invalid_price_returns_zero():
    assert kelly_fraction(0.5, 0.0) == 0.0
    assert kelly_fraction(0.5, 1.0) == 0.0
    print("PASS: kelly_fraction handles invalid prices safely")


def test_recommend_position_picks_correct_side():
    rec = recommend_position(model_prob=0.65, market_yes_price=0.50)
    assert rec["side"] == "UP", f"expected UP recommendation, got {rec['side']}"
    assert rec["kelly_fraction"] > 0
    print("PASS: recommend_position picks the side with real edge")


def test_brier_score_perfect_prediction():
    score = brier_score([1.0, 0.0], [1, 0])
    assert score == 0.0, f"perfect predictions should score 0, got {score}"
    print("PASS: brier_score is 0 for perfect predictions")


def test_brier_score_always_50_50():
    score = brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
    assert abs(score - 0.25) < 1e-9, f"always-50% should score 0.25, got {score}"
    print("PASS: brier_score is 0.25 for always-guessing-50%")


def test_log_loss_matches_known_value():
    import math
    score = log_loss([0.5], [1])
    assert abs(score - (-math.log(0.5))) < 1e-9
    print("PASS: log_loss matches known closed-form value")


def test_max_drawdown_detects_real_drawdown():
    dd = max_drawdown([100, 120, 80, 90])
    assert abs(dd - (1 - 80/120)) < 1e-9, f"expected ~33% drawdown, got {dd}"
    print("PASS: max_drawdown correctly measures peak-to-trough decline")


def test_bootstrap_ci_rejects_noise():
    random.seed(1)
    preds = [random.random() for _ in range(200)]
    outcomes = [1 if random.random() < 0.5 else 0 for _ in range(200)]
    result = bootstrap_accuracy_ci(preds, outcomes)
    assert not result["distinguishable_from_coinflip"], \
        "pure noise should NOT be flagged as distinguishable from a coin flip"
    print("PASS: bootstrap CI correctly rejects pure noise as having no edge")


def test_bootstrap_ci_detects_real_signal():
    random.seed(2)
    preds, outcomes = [], []
    for _ in range(500):
        actual = 1 if random.random() < 0.5 else 0
        correct = random.random() < 0.60
        pred = (0.7 if actual == 1 else 0.3) if correct else (0.3 if actual == 1 else 0.7)
        preds.append(pred)
        outcomes.append(actual)
    result = bootstrap_accuracy_ci(preds, outcomes)
    assert result["distinguishable_from_coinflip"], \
        "a genuine 60% signal at n=500 should be flagged as distinguishable"
    print("PASS: bootstrap CI correctly detects a genuine signal")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests)-failed}/{len(tests)} tests passed")
    if failed:
        sys.exit(1)
