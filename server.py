"""
server.py — dashboard + API for the research platform. Same auth pattern
as the execution bot: refuses to start without DASHBOARD_USER/PASSWORD set.
"""
import os
import sys
import secrets
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database import db
from config.settings import load_weights
from backtesting.engine import run_full_backtest
from backtesting.monte_carlo import simulate_equity_paths
from models.kelly import recommend_position
from backtesting.metrics import sortino_ratio, profit_factor, max_drawdown, check_promotion
import scan_markets

DASHBOARD_USER = os.environ.get("DASHBOARD_USER")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD")
if not DASHBOARD_USER or not DASHBOARD_PASSWORD:
    raise RuntimeError("DASHBOARD_USER and DASHBOARD_PASSWORD must be set in .env before starting.")

security = HTTPBasic()
START_TIME = time.time()


def require_auth(credentials: HTTPBasicCredentials = Depends(security)):
    ok_user = secrets.compare_digest(credentials.username, DASHBOARD_USER)
    ok_pass = secrets.compare_digest(credentials.password, DASHBOARD_PASSWORD)
    if not (ok_user and ok_pass):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                             detail="Invalid credentials", headers={"WWW-Authenticate": "Basic"})
    return credentials.username


app = FastAPI()
db.init_db()


class MonteCarloRequest(BaseModel):
    win_prob: float
    market_price: float = 0.5
    kelly_multiplier: float = 0.5
    n_trades: int = 200
    n_simulations: int = 1000


@app.get("/")
def index(user: str = Depends(require_auth)):
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "dashboard.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - START_TIME)}


@app.get("/api/backtests")
def backtests(user: str = Depends(require_auth)):
    return db.latest_backtest_runs(limit=30)


@app.get("/api/predictions")
def predictions(user: str = Depends(require_auth)):
    return db.recent_predictions(limit=50)


@app.post("/api/run-backtest")
def trigger_backtest(n_windows: int = 100, user: str = Depends(require_auth)):
    return run_full_backtest(n_windows=n_windows)


@app.post("/api/run-scan")
def trigger_scan(user: str = Depends(require_auth)):
    return scan_markets.run_scan()


@app.post("/api/run-montecarlo")
def trigger_montecarlo(req: MonteCarloRequest, user: str = Depends(require_auth)):
    return simulate_equity_paths(
        win_prob=req.win_prob, market_price=req.market_price,
        kelly_multiplier=req.kelly_multiplier, n_trades=req.n_trades,
        n_simulations=req.n_simulations,
    )


@app.get("/api/weights")
def weights(user: str = Depends(require_auth)):
    return load_weights()


@app.get("/api/alpha")
def alpha(user: str = Depends(require_auth)):
    """
    THE ANSWER TO 'accuracy alone is misleading': for every resolved
    prediction, simulate the Kelly-sized paper bet it would have placed
    (only when there was actual edge) and compute the real return. Reports
    both the simple sum (easy to read) and the geometric mean per bet (the
    mathematically correct "true average return" when bets compound —
    arithmetic mean overstates true growth whenever returns vary, which
    they always do). Rule of 72 turns that geometric mean into an intuitive
    "how many bets to double the bankroll" estimate.
    """
    resolved = db.resolved_predictions(limit=1000)
    bets_taken = 0
    cumulative_return = 0.0
    compound_growth = 1.0
    wins = 0
    curve = [0.0]
    per_bet_returns = []

    for row in resolved:
        rec = recommend_position(row["model_prob"], row["market_prob"])
        if rec["side"] is None:
            continue  # no edge at prediction time -> no bet would have been placed
        bets_taken += 1
        actual_up = bool(row["outcome_up"])
        if rec["side"] == "UP":
            entry_price = row["market_prob"]
            won = actual_up
        else:
            entry_price = 1 - row["market_prob"]
            won = not actual_up
        entry_price = max(min(entry_price, 0.99), 0.01)
        frac = rec["kelly_fraction"]
        pnl = frac * (1 - entry_price) / entry_price if won else -frac
        cumulative_return += pnl
        compound_growth *= (1 + pnl)
        per_bet_returns.append(pnl)
        if won:
            wins += 1
        curve.append(cumulative_return)

    geometric_mean_return = None
    bets_to_double = None
    if bets_taken > 0 and compound_growth > 0:
        geometric_mean_return = compound_growth ** (1 / bets_taken) - 1
        if geometric_mean_return > 0:
            bets_to_double = 72 / (geometric_mean_return * 100)  # Rule of 72

    sortino = sortino_ratio(per_bet_returns) if per_bet_returns else None
    pf = profit_factor(per_bet_returns) if per_bet_returns else None
    dd = max_drawdown([1.0 + c for c in curve]) if len(curve) > 1 else None

    from database import db as _db
    latest_bt = _db.latest_backtest_runs(limit=5)
    distinguishable = None
    for r in latest_bt:
        import json as _json
        feat = _json.loads(r.get("per_feature_accuracy_json") or "{}")
        cm = feat.get("combined_model", {})
        if cm.get("distinguishable_from_coinflip") is not None:
            distinguishable = cm["distinguishable_from_coinflip"]
            break

    promoted, promotion_failures = check_promotion(bets_taken, distinguishable, sortino, dd, pf)

    return {
        "n_resolved_total": len(resolved),
        "bets_taken": bets_taken,
        "bets_skipped_no_edge": len(resolved) - bets_taken,
        "win_rate_of_bets_taken": (wins / bets_taken) if bets_taken else None,
        "cumulative_return_fraction": cumulative_return,
        "compound_growth_multiple": compound_growth if bets_taken else None,
        "geometric_mean_return_per_bet": geometric_mean_return,
        "bets_to_double_rule_of_72": bets_to_double,
        "sortino_ratio": sortino,
        "profit_factor": pf,
        "max_drawdown": dd,
        "promoted": promoted,
        "promotion_failures": promotion_failures,
        "equity_curve": curve,
    }


@app.get("/api/kpi")
def kpi(user: str = Depends(require_auth)):
    preds = db.recent_predictions(limit=1)
    backtests = db.latest_backtest_runs(limit=10)
    all_preds = db.recent_predictions(limit=1000)
    collection_stats = db.data_collection_stats()
    return {
        "last_scan_ts": preds[0]["ts"] if preds else None,
        "total_predictions": len(all_preds),
        "last_backtest_ts": backtests[0]["run_at"] if backtests else None,
        "timeframes_tracked": len(set(b["timeframe"] for b in backtests)),
        "total_resolved": collection_stats["total_resolved"],
        "days_collecting": collection_stats["days_collecting"],
    }


@app.get("/api/accuracy")
def accuracy(user: str = Depends(require_auth)):
    """Live accuracy view (#2): of predictions that have since resolved,
    how often did the MODEL (not just individual features) call it right?
    This is the number that actually tells you if this is working."""
    resolved = db.resolved_predictions(limit=500)
    if not resolved:
        return {"n": 0, "accuracy": None, "message": "no resolved predictions yet"}
    correct = sum(1 for r in resolved if (r["model_prob"] >= 0.5) == bool(r["outcome_up"]))
    n = len(resolved)
    by_timeframe = {}
    for r in resolved:
        tf = r["timeframe"]
        by_timeframe.setdefault(tf, {"correct": 0, "n": 0})
        by_timeframe[tf]["n"] += 1
        if (r["model_prob"] >= 0.5) == bool(r["outcome_up"]):
            by_timeframe[tf]["correct"] += 1
    return {
        "n": n,
        "accuracy": correct / n,
        "by_timeframe": {tf: {"accuracy": s["correct"]/s["n"], "n": s["n"]} for tf, s in by_timeframe.items()},
    }


@app.get("/api/feedback")
def feedback(user: str = Depends(require_auth)):
    return db.latest_feedback_run()


@app.post("/api/test-alert")
def test_alert(user: str = Depends(require_auth)):
    """#3: one-tap way to confirm Telegram is actually wired up and a
    message really arrives on your phone, instead of just trusting the code."""
    from telegram import notifier
    if not notifier.is_configured():
        return {"sent": False, "reason": "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env"}
    notifier.send_message("✅ Test alert from BTC Research Platform — if you see this, Telegram is working.")
    return {"sent": True}


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
