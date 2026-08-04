"""
backtesting/monte_carlo.py — two distinct, deliberately separate uses:

1. BOOTSTRAP CONFIDENCE INTERVAL (the important one right now): resamples
   the historical backtest outcomes thousands of times to answer "how
   confident can we be that this accuracy is different from a coin flip?"
   This is exactly the statistical question that a 100-vs-500-window
   backtest earlier in this project answered the hard way (an apparent
   edge evaporated at a larger sample). Bootstrap CI gives that answer
   directly, with an explicit interval, instead of needing to guess how
   big a sample is "big enough."

2. FORWARD EQUITY SIMULATION: given an assumed edge and Kelly sizing,
   simulates many possible future trading paths to show the realistic
   RANGE of outcomes (including risk of ruin) — not a single expected-value
   number. Only meaningful once real edge exists; right now it's built and
   tested, ready for when that happens.
"""
import random
import math


def bootstrap_accuracy_ci(predictions, outcomes, n_bootstrap=2000, ci=0.95):
    """
    Resamples (prediction, outcome) pairs with replacement n_bootstrap times,
    computing accuracy each time. Returns the empirical confidence interval
    and a direct answer to "is this distinguishable from 50%?"
    """
    n = len(predictions)
    if n < 10:
        return {"error": f"only {n} samples — need at least 10 for a meaningful bootstrap"}

    paired = list(zip(predictions, outcomes))
    accuracies = []
    for _ in range(n_bootstrap):
        sample = [paired[random.randrange(n)] for _ in range(n)]
        correct = sum(1 for p, o in sample if (p >= 0.5) == (o == 1))
        accuracies.append(correct / n)

    accuracies.sort()
    lower_idx = int((1 - ci) / 2 * n_bootstrap)
    upper_idx = int((1 - (1 - ci) / 2) * n_bootstrap) - 1
    lower = accuracies[lower_idx]
    upper = accuracies[upper_idx]
    mean_acc = sum(accuracies) / len(accuracies)

    # fraction of bootstrap resamples that were AT or BELOW 50% —
    # a rough one-sided p-value-like read on "is this really better than a coin flip"
    frac_at_or_below_half = sum(1 for a in accuracies if a <= 0.5) / n_bootstrap

    return {
        "n_samples": n,
        "point_estimate_accuracy": sum(1 for p, o in paired if (p >= 0.5) == (o == 1)) / n,
        "bootstrap_mean_accuracy": mean_acc,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": ci,
        "fraction_at_or_below_50pct": frac_at_or_below_half,
        "distinguishable_from_coinflip": lower > 0.5,  # the honest bar to clear
    }


def simulate_equity_paths(win_prob, market_price, kelly_multiplier=0.5,
                           n_trades=200, n_simulations=1000, starting_equity=1.0):
    """
    Simulates n_simulations independent paths of n_trades each, betting the
    Kelly-recommended fraction every time, assuming win_prob and market_price
    stay constant (a simplification — real edge/price move around, this is
    a stylized "if the edge holds" simulation, not a forecast).
    """
    from .metrics import max_drawdown as _max_dd
    from models.kelly import kelly_fraction

    f = kelly_fraction(win_prob, market_price) * kelly_multiplier
    f = max(0.0, min(1.0, f))

    if f <= 0:
        return {"error": "no positive Kelly fraction at this win_prob/price — nothing to simulate"}

    final_equities = []
    max_drawdowns = []
    ruin_count = 0
    RUIN_THRESHOLD = 0.1  # equity falling below 10% of start counts as "ruin" for this purpose

    for _ in range(n_simulations):
        equity = starting_equity
        curve = [equity]
        ruined = False
        for _ in range(n_trades):
            stake = equity * f
            if random.random() < win_prob:
                # win: shares bought at market_price pay $1 each
                equity += stake * (1 - market_price) / market_price
            else:
                equity -= stake
            equity = max(equity, 0)
            curve.append(equity)
            if equity <= starting_equity * RUIN_THRESHOLD:
                ruined = True
        final_equities.append(equity)
        max_drawdowns.append(_max_dd(curve))
        if ruined:
            ruin_count += 1

    final_equities.sort()
    n = len(final_equities)
    return {
        "win_prob": win_prob,
        "market_price": market_price,
        "kelly_fraction_used": f,
        "n_simulations": n_simulations,
        "n_trades_per_sim": n_trades,
        "median_final_equity": final_equities[n // 2],
        "p10_final_equity": final_equities[int(n * 0.10)],
        "p90_final_equity": final_equities[int(n * 0.90)],
        "mean_max_drawdown": sum(max_drawdowns) / len(max_drawdowns),
        "probability_of_ruin": ruin_count / n_simulations,
    }
