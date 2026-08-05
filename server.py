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
