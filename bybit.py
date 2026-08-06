"""
data_collection/bybit.py — free, public, no API key. Used alongside
Binance's funding rate to build a genuine cross-exchange divergence signal
that a single-exchange view can't see.
"""
import requests

BASE = "https://api.bybit.com/v5/market"


def get_funding_rate_history(symbol="BTCUSDT", start_time_ms=None, end_time_ms=None, limit=200):
    """Bybit requires startTime and endTime together, or neither — never just one."""
    params = {"category": "linear", "symbol": symbol, "limit": limit}
    if start_time_ms is not None and end_time_ms is not None:
        params["startTime"] = start_time_ms
        params["endTime"] = end_time_ms
    resp = requests.get(f"{BASE}/funding/history", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("result", {}).get("list", [])  # [{symbol, fundingRate, fundingRateTimestamp}, ...]
