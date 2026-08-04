"""
data_collection/binance.py — free Binance spot + futures data, no API key
needed. Historical funding rate and open interest ARE actually available
here (unlike Polymarket's historical order books, which aren't retained
anywhere free) — that's why external_data can be properly backtested while
order_flow_volume and market_probability mostly can't be.
"""
import time
import requests

SPOT_BASE = "https://api.binance.com/api/v3"
FUTURES_BASE = "https://fapi.binance.com/fapi/v1"
FUTURES_DATA_BASE = "https://fapi.binance.com/futures/data"


def fetch_klines_paginated(symbol, interval, total_needed, end_time_ms=None):
    out = []
    end_time = end_time_ms
    remaining = total_needed
    while remaining > 0:
        batch = min(1000, remaining)
        params = {"symbol": symbol, "interval": interval, "limit": batch}
        if end_time is not None:
            params["endTime"] = end_time
        resp = requests.get(f"{SPOT_BASE}/klines", params=params, timeout=15)
        resp.raise_for_status()
        candles = resp.json()
        if not candles:
            break
        out = candles + out
        end_time = candles[0][0] - 1
        remaining -= len(candles)
        if len(candles) < batch:
            break
        time.sleep(0.15)
    return out


def get_spot_ticker(symbol):
    resp = requests.get(f"{SPOT_BASE}/ticker/24hr", params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_funding_rate_history(symbol, start_time_ms=None, end_time_ms=None, limit=1000):
    """Free, no key. Retention is limited (~a few months) but plenty for
    validating a short-horizon hypothesis."""
    params = {"symbol": symbol, "limit": limit}
    if start_time_ms:
        params["startTime"] = start_time_ms
    if end_time_ms:
        params["endTime"] = end_time_ms
    resp = requests.get(f"{FUTURES_BASE}/fundingRate", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()  # [{fundingTime, fundingRate, symbol}, ...]


def get_open_interest_now(symbol):
    resp = requests.get(f"{FUTURES_BASE}/openInterest", params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_open_interest_history(symbol, period="5m", limit=500):
    """Historical OI retention is short (~30 days) on Binance's free endpoint —
    fine for an initial validation pass, not for years of backtesting."""
    resp = requests.get(f"{FUTURES_DATA_BASE}/openInterestHist",
                         params={"symbol": symbol, "period": period, "limit": limit}, timeout=15)
    resp.raise_for_status()
    return resp.json()
