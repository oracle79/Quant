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

FEATURE_INTERVAL = {"5m": "1m", "15m": "1m", "1h": "5m", "4h": "15m", "1d": "1h"}
LOOKBACK = {"5m": 30, "15m": 45, "1h": 36, "4h": 32, "1d": 48}


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
        start_date=market.get("startDate"), end_date=market.get("endDate"),
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


def run_scan():
    db.init_db()
    scanned = []
    # first pass: get each asset's price per timeframe so cross_market has something to reference
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
            except Exception as e:
                log.exception(f"[{asset} {tf}] scan failed: {e}")

    report = format_morning_report(scanned)
    log.info(report)
    notifier.send_message(report)
    return scanned


if __name__ == "__main__":
    run_scan()
