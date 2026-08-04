"""
models/kelly.py — Kelly criterion for a binary share priced 0-1, paying $1
if correct. f* = p - (1-p) * price / (1-price). Same formula validated in
the execution bot; centralized here so both the live scanner and the
Monte Carlo simulator use one definition.
"""

def kelly_fraction(p, price):
    if price <= 0 or price >= 1:
        return 0.0
    return p - (1 - p) * price / (1 - price)


def recommend_position(model_prob, market_yes_price, kelly_multiplier=0.5):
    """
    Returns the recommended side and Kelly fraction, using whichever side
    (Up/Down) actually shows positive edge. kelly_multiplier defaults to
    half-Kelly — full Kelly is provably correct in the idealized case but
    assumes your probability estimate is exactly right, which it never is
    in practice; half-Kelly trades some growth rate for a lot less variance.
    """
    market_no_price = 1 - market_yes_price
    model_prob_no = 1 - model_prob

    f_up = kelly_fraction(model_prob, market_yes_price)
    f_down = kelly_fraction(model_prob_no, market_no_price)

    if f_up <= 0 and f_down <= 0:
        return {"side": None, "kelly_fraction": 0.0, "raw_kelly_fraction": 0.0}

    if f_up > f_down:
        side, frac = "UP", f_up
    else:
        side, frac = "DOWN", f_down

    frac = min(frac, 1.0)
    return {
        "side": side,
        "kelly_fraction": frac * kelly_multiplier,
        "raw_kelly_fraction": frac,  # full Kelly, before the safety multiplier
    }
