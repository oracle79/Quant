"""
backtesting/metrics.py — the actual measurements the spec requires.
Kept separate from the engine so each metric is independently testable
and reusable (e.g. the feedback loop can reuse brier_score directly).
"""
import math


def brier_score(predictions, outcomes):
    """Mean squared error between predicted probability and actual outcome (0/1).
    Lower is better. 0 = perfect. 0.25 = what always-guess-50% scores."""
    n = len(predictions)
    if n == 0:
        return None
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n


def log_loss(predictions, outcomes, eps=1e-9):
    n = len(predictions)
    if n == 0:
        return None
    total = 0.0
    for p, o in zip(predictions, outcomes):
        p = max(min(p, 1 - eps), eps)
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / n


def calibration_bins(predictions, outcomes, n_bins=10):
    """Returns [(bin_center, predicted_avg, actual_avg, n_in_bin), ...].
    A well-calibrated model has predicted_avg ~= actual_avg in every bin."""
    bins = [[] for _ in range(n_bins)]
    for p, o in zip(predictions, outcomes):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, o))
    result = []
    for i, b in enumerate(bins):
        if not b:
            continue
        center = (i + 0.5) / n_bins
        pred_avg = sum(x[0] for x in b) / len(b)
        actual_avg = sum(x[1] for x in b) / len(b)
        result.append((center, pred_avg, actual_avg, len(b)))
    return result


def sharpe_ratio(returns, risk_free=0.0):
    """returns: list of per-trade or per-period returns (fractions, e.g. 0.02 = +2%)."""
    n = len(returns)
    if n < 2:
        return None
    mean_ret = sum(returns) / n
    variance = sum((r - mean_ret) ** 2 for r in returns) / n
    sd = math.sqrt(variance)
    if sd == 0:
        return None
    return (mean_ret - risk_free) / sd * math.sqrt(n)


def max_drawdown(equity_curve):
    """equity_curve: list of cumulative equity values over time."""
    if not equity_curve:
        return None
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    return max_dd
