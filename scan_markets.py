"""
scan_markets.py — the live orchestration script. Run on a schedule (see
deploy/ for systemd timer). Scans every active BTC/ETH market across all
timeframes, runs the full model, logs the prediction (append-only, never
overwritten), and sends a ranked report via Telegram.

This does NOT execute trades — per spec, this is a research and alert
platform. Execution is explicitly a future phase.
"""
import sys
import os
import time
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import CRYPTO_ASSETS, TIMEFRAMES, BINANCE_SYMBOLS, load_weights
from data_collection import polymarket, binance
from models.probability_model import run_model
from reports.report_generator import format_morning_report
from telegram import notifier
from database import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scan_markets")

_last_error_alert = {}
ERROR_ALERT_THROTTLE_SEC = 1800  # don't re-alert the same failure more than once per 30 min

def _maybe_alert_error(context, error):
    now = time.time()
    if now - _last_error_alert.get(context, 0) > ERROR_ALERT_THROTTLE_SEC:
        notifier.send_message(f"⚠️ Research platform error in {context}: {error}")
        _last_error_alert[context] = now

FEATURE_INTERVAL = {"5m": "1m", "15m": "1m", "1h": "5m", "4h": "15m", "1d": "1h"}
LOOKBACK = {"5m": 30, "15m": 45, "1h": 36, "4h": 32, "1d": 48}


def _parse_iso_to_epoch(iso_str):
    if not iso_str:
        return None
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def scan_one(asset, timeframe, correlated_prices):
    market = polymarket.find_current_market(asset, timeframe)
    if not market:
        return None

    prices = polymarket.parse_prices(market)
    token_ids = polymarket.parse_token_ids(market)
    if len(prices) < 2:
        return None
    yes_price = prices[0]

    market_id = db.upsert_market(
        slug=market.get("slug"), question=market.get("question"), category="crypto",
        asset=asset, timeframe=timeframe,
        start_date=_parse_iso_to_epoch(market.get("startDate")),
        end_date=_parse_iso_to_epoch(market.get("endDate")),
    )

    liquidity = float(market.get("liquidity") or 0)
    volume_24hr = float(market.get("volume24hr") or 0)

    book_metrics = {}
    if len(token_ids) >= 1:
        try:
            book = polymarket.get_order_book(token_ids[0])
            book_metrics = polymarket.book_metrics(book)
        except Exception as e:
            log.warning(f"[{asset} {timeframe}] order book fetch failed: {e}")

    db.insert_market_snapshot(
        market_id, yes_price, prices[1] if len(prices) > 1 else None,
        volume_24hr, liquidity,
        book_metrics.get("best_bid"), book_metrics.get("best_ask"), book_metrics.get("ask_depth_usd"),
    )

    # underlying asset data
    symbol = BINANCE_SYMBOLS[asset]
    kl = binance.fetch_klines_paginated(symbol, FEATURE_INTERVAL[timeframe], LOOKBACK[timeframe])
    closes = [float(c[4]) for c in kl]

    try:
        funding_raw = binance.get_funding_rate_history(symbol, limit=30)
        funding_hist = [float(f["fundingRate"]) for f in funding_raw]
    except Exception:
        funding_hist = []
    try:
        oi_raw = binance.get_open_interest_history(symbol, period="5m", limit=30)
        oi_hist = [float(o["sumOpenInterest"]) for o in oi_raw]
    except Exception:
        oi_hist = []

    context = {
        "yes_price": yes_price,
        "liquidity": liquidity,
        "spread": book_metrics.get("spread"),
        "best_bid": book_metrics.get("best_bid"),
        "best_ask": book_metrics.get("best_ask"),
        "bid_depth_usd": book_metrics.get("bid_depth_usd", 0),
        "ask_depth_usd": book_metrics.get("ask_depth_usd", 0),
        "closes": closes,
        "funding_rate_history": funding_hist,
        "open_interest_history": oi_hist,
        "correlated_asset_yes_price": correlated_prices.get(timeframe),
        "correlated_asset_name": "ETH" if asset == "BTC" else "BTC",
        "asset": asset,
        "timeframe": timeframe,
        "db_module": db,
    }

    result = run_model(context)
    db.insert_prediction(
        market_id, result["market_prob"], result["model_prob"], result["edge"],
        result["confidence"], {k: v for k, v in result["feature_results"].items()},
        "; ".join(f"{n}: {e}" for n, e in result["top_drivers"]),
    )

    return {
        "question": market.get("question"), "asset": asset, "timeframe": timeframe,
        "market_prob": result["market_prob"], "model_prob": result["model_prob"],
        "edge": result["edge"], "confidence": result["confidence"],
        "liquidity": liquidity, "top_drivers": result["top_drivers"],
    }


BIG_EDGE_THRESHOLD_PP = float(os.environ.get("BIG_EDGE_THRESHOLD_PP", "5.0"))
ALERT_STATE_PATH = os.environ.get("EDGE_ALERT_STATE_PATH", "edge_alert_state.json")
ALERT_REPEAT_THROTTLE_SEC = 3600  # don't re-alert the same market more than once/hour


def _load_alert_state():
    import json
    if os.path.exists(ALERT_STATE_PATH):
        try:
            with open(ALERT_STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_alert_state(state):
    import json
    with open(ALERT_STATE_PATH, "w") as f:
        json.dump(state, f)


def maybe_alert_big_edge(result, alert_state):
    edge_pp = result["edge"] * 100
    if abs(edge_pp) < BIG_EDGE_THRESHOLD_PP:
        return
    slug_key = f"{result['asset']}_{result['timeframe']}"
    last_alert = alert_state.get(slug_key, 0)
    if time.time() - last_alert < ALERT_REPEAT_THROTTLE_SEC:
        return  # already alerted on this one recently, don't spam

    direction = "UP" if result["edge"] > 0 else "DOWN"
    drivers = "; ".join(f"{n}: {e}" for n, e in result.get("top_drivers", [])[:2])
    notifier.send_message(
        f"🚨 Big edge — {result['asset']} {result['timeframe']} ({direction})\n"
        f"Market: {result['market_prob']*100:.1f}% | Model: {result['model_prob']*100:.1f}% | "
        f"Edge: {edge_pp:+.1f}pp | Confidence: {result['confidence']*100:.0f}%\n{drivers}"
    )
    alert_state[slug_key] = time.time()


def check_and_resolve_markets():
    """THE CRITICAL FIX: without this, resolutions never gets populated, and
    historical_base_rate (plus any future accuracy scoring) can never work.
    Checks every market whose window has closed but has no recorded outcome
    yet, and asks Polymarket directly whether it resolved."""
    pending = db.markets_needing_resolution()
    resolved_count = 0
    for m in pending:
        try:
            market = polymarket.get_market_by_slug(m["slug"])
        except Exception:
            continue
        if not market or not market.get("closed"):
            continue
        prices = polymarket.parse_prices(market)
        if len(prices) < 2:
            continue
        outcome_up = round(prices[0])  # resolved markets settle to 0.0 or 1.0
        db.insert_resolution(m["id"], outcome_up)
        resolved_count += 1
    if resolved_count:
        log.info(f"resolved {resolved_count} markets")
    return resolved_count


def run_scan():
    db.init_db()
    scanned = []
    alert_state = _load_alert_state()
    correlated = {"BTC": {}, "ETH": {}}
    for asset in CRYPTO_ASSETS:
        for tf in TIMEFRAMES:
            m = polymarket.find_current_market(asset, tf)
            if m:
                prices = polymarket.parse_prices(m)
                if prices:
                    correlated[asset][tf] = prices[0]

    for asset in CRYPTO_ASSETS:
        other = "ETH" if asset == "BTC" else "BTC"
        for tf in TIMEFRAMES:
            try:
                result = scan_one(asset, tf, correlated[other])
                if result:
                    scanned.append(result)
                    maybe_alert_big_edge(result, alert_state)
            except Exception as e:
                log.exception(f"[{asset} {tf}] scan failed: {e}")
                _maybe_alert_error(f"scan[{asset} {tf}]", e)

    _save_alert_state(alert_state)

    try:
        check_and_resolve_markets()
    except Exception as e:
        log.exception(f"resolution check failed: {e}")
        _maybe_alert_error("check_and_resolve_markets", e)

    report = format_morning_report(scanned)
    log.info(report)
    return scanned


def send_daily_digest():
    """Separate from run_scan() on purpose — run_scan() now runs every 15
    minutes for data collection + big-edge alerts, and sending the FULL
    report that often would be spam. This sends one digest per day."""
    scanned = run_scan()
    report = format_morning_report(scanned)
    notifier.send_message(report)
    return scanned


if __name__ == "__main__":
    try:
        if "--digest" in sys.argv:
            send_daily_digest()
        else:
            run_scan()
    except Exception as e:
        log.exception(f"scan run failed entirely: {e}")
        notifier.send_message(f"🛑 Research platform: scheduled scan crashed entirely: {e}")
        raise
