"""
data_collection/polymarket.py — live market + order book fetching,
generalized across assets (BTC, ETH) and timeframes. Reuses the
Eastern-Time-aware slug logic already validated today (correct across
both DST regimes).
"""
import json
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
ET = ZoneInfo("America/New_York")
MONTHS = ["january","february","march","april","may","june","july","august",
          "september","october","november","december"]


def _window_start(now_sec, period_sec):
    return now_sec - (now_sec % period_sec)


def _get_et_parts(dt_utc):
    dt_et = dt_utc.astimezone(ET)
    return {"year": dt_et.year, "month": MONTHS[dt_et.month-1], "day": dt_et.day,
            "hour12": dt_et.hour % 12 or 12, "ampm": "am" if dt_et.hour < 12 else "pm"}


def slug_candidates_for(asset, timeframe, now_utc=None):
    now_utc = now_utc or datetime.now(ZoneInfo("UTC"))
    asset_prefix = asset.lower()  # 'btc' or 'eth'
    if timeframe in ("5m", "15m", "4h"):
        period = {"5m": 300, "15m": 900, "4h": 14400}[timeframe]
        ws = _window_start(int(now_utc.timestamp()), period)
        return [f"{asset_prefix}-updown-{timeframe}-{ws}", f"{asset_prefix}-updown-{timeframe}-{ws+period}"]
    elif timeframe == "1h":
        p = _get_et_parts(now_utc)
        name = "bitcoin" if asset == "BTC" else "ethereum"
        return [
            f"{name}-up-or-down-{p['month']}-{p['day']}-{p['year']}-{p['hour12']}{p['ampm']}-et",
            f"{name}-up-or-down-{p['month']}-{p['day']}-{p['hour12']}{p['ampm']}-et",
        ]
    elif timeframe == "1d":
        from datetime import timedelta
        dt_et = now_utc.astimezone(ET)
        settlement = dt_et.date() if dt_et.hour < 12 else (dt_et + timedelta(days=1)).date()
        name = "bitcoin" if asset == "BTC" else "ethereum"
        month_name = MONTHS[settlement.month-1]
        return [f"{name}-up-or-down-on-{month_name}-{settlement.day}-{settlement.year}"]
    raise ValueError(f"unknown timeframe {timeframe}")


def fetch_json(url, params=None):
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_market_by_slug(slug):
    data = fetch_json(f"{GAMMA}/markets/slug/{slug}")
    return data[0] if isinstance(data, list) else data


def find_current_market(asset, timeframe):
    for slug in slug_candidates_for(asset, timeframe):
        try:
            m = get_market_by_slug(slug)
            if m and not m.get("closed"):
                return m
        except Exception:
            continue
    return None


def parse_prices(market):
    raw = market.get("outcomePrices")
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [float(p) for p in (raw or [])]


def parse_token_ids(market):
    raw = market.get("clobTokenIds")
    if isinstance(raw, str):
        raw = json.loads(raw)
    return raw or []


def get_order_book(token_id):
    return fetch_json(f"{CLOB}/book", params={"token_id": token_id})


def book_metrics(book):
    bids = sorted([{"price": float(b["price"]), "size": float(b["size"])} for b in book.get("bids", [])],
                  key=lambda x: -x["price"])
    asks = sorted([{"price": float(a["price"]), "size": float(a["size"])} for a in book.get("asks", [])],
                  key=lambda x: x["price"])
    best_bid = bids[0]["price"] if bids else None
    best_ask = asks[0]["price"] if asks else None
    bid_depth = sum(b["price"]*b["size"] for b in bids if best_bid is not None and abs(b["price"]-best_bid) <= 0.03)
    ask_depth = sum(a["price"]*a["size"] for a in asks if best_ask is not None and abs(a["price"]-best_ask) <= 0.03)
    spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
    return {"best_bid": best_bid, "best_ask": best_ask, "bid_depth_usd": bid_depth,
            "ask_depth_usd": ask_depth, "spread": spread}
