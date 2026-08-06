"""
config/settings.py — all tunable parameters in one place, per the "never
overwrite historical data, make everything measurable" philosophy. Weights
are the STARTING HYPOTHESIS from the spec, not fixed truth — the whole
point of the backtesting engine is to test whether they deserve to stay
at these values.
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("RESEARCH_DB_PATH", os.path.join(BASE_DIR, "research.db"))
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", os.path.join(BASE_DIR, "config", "weights.json"))

# Starting hypothesis weights, exactly as specified. These get validated —
# not trusted — by the backtesting engine.
DEFAULT_WEIGHTS = {
    "market_probability":  0.405,
    "historical_base_rate":0.135,
    "external_data":       0.135,
    "order_flow_volume":   0.090,
    "momentum":            0.0675,
    "cross_market":        0.045,
    "market_efficiency":   0.0225,
    "cross_exchange_funding": 0.10,
}

# Crypto assets covered. Politics/macro categories are intentionally out of
# scope for this build — see spec discussion. Perps are a planned extension
# once Polymarket's perp markets stabilize; treated as a future data source,
# not built into v1.
# BTC only for now — the biggest, most liquid crypto asset on Polymarket,
# and the one most likely to eventually support futures/perp execution.
CRYPTO_ASSETS = ["BTC"]
BINANCE_SYMBOLS = {"BTC": "BTCUSDT"}

TIMEFRAMES = {
    "5m":  {"window_sec": 300,   "feature_interval": "1m", "lookback": 30},
    "15m": {"window_sec": 900,   "feature_interval": "1m", "lookback": 45},
    "1h":  {"window_sec": 3600,  "feature_interval": "5m", "lookback": 36},
    "4h":  {"window_sec": 14400, "feature_interval": "15m","lookback": 32},
    "1d":  {"window_sec": 86400, "feature_interval": "1h", "lookback": 48},
}


def load_weights():
    if os.path.exists(WEIGHTS_PATH):
        try:
            with open(WEIGHTS_PATH) as f:
                w = json.load(f)
            if set(w.keys()) == set(DEFAULT_WEIGHTS.keys()):
                return w
        except Exception:
            pass
    return dict(DEFAULT_WEIGHTS)


def save_weights(weights):
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)
