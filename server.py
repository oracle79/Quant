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


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
