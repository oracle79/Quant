"""
backtesting/engine.py — replays historical crypto up/down windows through
the SAME feature/model code that runs live (per spec: "every feature must
prove that it improves predictive performance before remaining in the
model"). No lookahead: every feature at window t only sees data strictly
before that window opens.

IMPORTANT HONEST LIMITATION: Polymarket doesn't retain historical
order-book or price data anywhere free/reliable, so market_probability,
order_flow_volume, cross_market, and historical_base_rate can't be
properly backtested against real historical Polymarket prices. This engine
sets yes_price=0.5 (neutral) during backtesting so market_probability
doesn't skew results, and reports each feature's OWN standalone directional
accuracy — the same rigorous approach validated in today's BTC backtest,
now generalized across the full feature set. Only external_data and
momentum get a real historical test right now, because Binance actually
retains that history for free. The rest will only become properly
backtestable once the platform has run live long enough to build its own
history (that's what market_snapshots/resolutions/predictions are for).
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TIMEFRAMES, BINANCE_SYMBOLS, load_weights
from data_collection.binance import fetch_klines_paginated, get_funding_rate_history, get_open_interest_history
from features.momentum import MomentumFeature
from features.external_data import ExternalDataFeature
from models.probability_model import run_model
from backtesting.metrics import brier_score, log_loss, calibration_bins
from backtesting.monte_carlo import bootstrap_accuracy_ci
from database import db

INTERVAL_SEC = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}


def backtest_asset_timeframe(asset, timeframe, n_windows=100):
    symbol = BINANCE_SYMBOLS[asset]
    cfg = TIMEFRAMES[timeframe]
    window_sec = cfg["window_sec"]
    interval = cfg["feature_interval"]
    lookback = cfg["lookback"]
    candles_per_window = max(1, window_sec // INTERVAL_SEC[interval])

    total_needed = (n_windows + 1) * candles_per_window + lookback + 5
    raw = fetch_klines_paginated(symbol, interval, total_needed)
    if len(raw) < candles_per_window * 2 + lookback:
        raise RuntimeError(f"not enough historical data for {asset} {timeframe}")

    closes = [float(c[4]) for c in raw]
    timestamps = [c[0] for c in raw]

    # funding rate + OI history covering the whole backtest span
    start_ms, end_ms = timestamps[0], timestamps[-1]
    try:
        funding_raw = get_funding_rate_history(symbol, start_time_ms=start_ms, end_time_ms=end_ms, limit=1000)
        funding_series = [(f["fundingTime"], float(f["fundingRate"])) for f in funding_raw]
    except Exception:
        funding_series = []
    try:
        oi_raw = get_open_interest_history(symbol, period="5m", limit=500)
        oi_series = [(o["timestamp"], float(o["sumOpenInterest"])) for o in oi_raw]
    except Exception:
        oi_series = []
    try:
        from data_collection.bybit import get_funding_rate_history as get_bybit_funding_history
        bybit_raw = get_bybit_funding_history(symbol, start_time_ms=start_ms, end_time_ms=end_ms, limit=200)
        bybit_series = [(int(f["fundingRateTimestamp"]), float(f["fundingRate"])) for f in bybit_raw]
    except Exception:
        bybit_series = []

    momentum_feat = MomentumFeature()
    external_feat = ExternalDataFeature()

    n_windows = min(n_windows, (len(raw) - lookback) // candles_per_window - 1)

    momentum_preds, external_preds, combined_preds, outcomes = [], [], [], []

    for i in range(n_windows):
        window_end_idx = len(raw) - (n_windows - i) * candles_per_window
        window_start_idx = window_end_idx - candles_per_window
        feat_start_idx = window_start_idx - lookback
        if feat_start_idx < 0 or window_start_idx < 1:
            continue

        fc = closes[feat_start_idx:window_start_idx]
        if len(fc) < 10:
            continue
        window_open_ts = timestamps[window_start_idx - 1]

        momentum_result = momentum_feat.compute({"closes": fc})

        funding_hist = [r for t, r in funding_series if t < window_open_ts][-30:]
        oi_hist = [v for t, v in oi_series if t < window_open_ts][-30:]
        bybit_hist = [r for t, r in bybit_series if t < window_open_ts][-30:]
        external_result = external_feat.compute({
            "funding_rate_history": funding_hist,
            "open_interest_history": oi_hist,
        })

        # combined full-model prediction: features needing data we don't have
        # historically (market price, order book, cross-market, base rate)
        # degrade gracefully to low-confidence neutral automatically — see
        # each feature's own missing-data handling. This is the actual
        # "run the whole algorithm" test, not just individual features.
        combined_context = {
            "yes_price": 0.5, "liquidity": 0, "spread": None,
            "best_bid": None, "best_ask": None, "bid_depth_usd": 0, "ask_depth_usd": 0,
            "closes": fc,
            "funding_rate_history": funding_hist, "open_interest_history": oi_hist,
            "binance_funding_history": funding_hist, "bybit_funding_history": bybit_hist,
            "correlated_asset_yes_price": None, "correlated_asset_name": None,
            "asset": asset, "timeframe": timeframe, "db_module": db,
        }
        combined_result = run_model(combined_context)

        window_open = closes[window_start_idx - 1]
        window_close = closes[window_end_idx - 1]
        actual_up = 1 if window_close >= window_open else 0

        momentum_preds.append(momentum_result.score)
        external_preds.append(external_result.score)
        combined_preds.append(combined_result["model_prob"])
        outcomes.append(actual_up)

    per_feature_accuracy = {}
    for name, preds in [("momentum", momentum_preds), ("external_data", external_preds),
                         ("combined_model", combined_preds)]:
        if not preds:
            continue
        correct = sum(1 for p, o in zip(preds, outcomes) if (p >= 0.5) == (o == 1))
        ci = bootstrap_accuracy_ci(preds, outcomes)
        per_feature_accuracy[name] = {
            "accuracy": correct / len(preds),
            "n": len(preds),
            "brier": brier_score(preds, outcomes),
            "log_loss": log_loss(preds, outcomes),
            "ci_lower": ci.get("ci_lower"),
            "ci_upper": ci.get("ci_upper"),
            "distinguishable_from_coinflip": ci.get("distinguishable_from_coinflip"),
        }

    return {
        "asset": asset,
        "timeframe": timeframe,
        "n_windows": len(outcomes),
        "per_feature_accuracy": per_feature_accuracy,
        "outcomes": outcomes,
    }


def run_full_backtest(n_windows=100):
    from config.settings import CRYPTO_ASSETS, load_weights
    summary = {}
    weights = load_weights()
    for asset in CRYPTO_ASSETS:
        for tf in TIMEFRAMES:
            key = f"{asset}_{tf}"
            try:
                result = backtest_asset_timeframe(asset, tf, n_windows)
                summary[key] = result
                for feat_name, stats in result["per_feature_accuracy"].items():
                    pass  # per-feature already in result; top-level backtest_run logged once below
                db.log_backtest_run(
                    category="crypto", asset=asset, timeframe=tf,
                    n_markets=result["n_windows"],
                    accuracy=result["per_feature_accuracy"].get("combined_model", {}).get("accuracy"),
                    brier=result["per_feature_accuracy"].get("combined_model", {}).get("brier"),
                    log_loss_val=result["per_feature_accuracy"].get("combined_model", {}).get("log_loss"),
                    sharpe=None, max_dd=None,
                    per_feature_accuracy=result["per_feature_accuracy"],
                    weights_used=weights,
                )
            except Exception as e:
                summary[key] = {"error": str(e)}
                try:
                    from telegram import notifier
                    notifier.send_message(f"⚠️ Research platform: backtest failed for {key}: {e}")
                except Exception:
                    pass
    return summary
