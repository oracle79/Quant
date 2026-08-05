"""
database/db.py — SQLite, append-only. Per spec: "never overwrite historical
data, every prediction must be reproducible." Nothing in here ever UPDATEs
a past row — corrections happen by inserting a new row, not editing history.
"""
import sqlite3
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS markets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE,
            question TEXT,
            category TEXT,
            asset TEXT,
            timeframe TEXT,
            start_date REAL,
            end_date REAL,
            first_seen_at REAL
        );

        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id INTEGER,
            ts REAL,
            yes_price REAL,
            no_price REAL,
            volume_24hr REAL,
            liquidity REAL,
            best_bid REAL,
            best_ask REAL,
            ask_depth_usd REAL
        );

        CREATE TABLE IF NOT EXISTS external_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT,
            ts REAL,
            spot_price REAL,
            funding_rate REAL,
            open_interest REAL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id INTEGER,
            ts REAL,
            market_prob REAL,
            model_prob REAL,
            edge REAL,
            confidence REAL,
            feature_scores_json TEXT,
            explanation TEXT
        );

        CREATE TABLE IF NOT EXISTS resolutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id INTEGER,
            resolved_at REAL,
            outcome_up INTEGER  -- 1 = up/yes, 0 = down/no
        );

        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at REAL,
            category TEXT,
            asset TEXT,
            timeframe TEXT,
            n_markets INTEGER,
            accuracy REAL,
            brier_score REAL,
            log_loss REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            per_feature_accuracy_json TEXT,
            weights_used_json TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at REAL,
            n_predictions_analyzed INTEGER,
            old_weights_json TEXT,
            new_weights_json TEXT,
            per_feature_accuracy_json TEXT,
            summary TEXT
        )
    """)
    conn.commit()
    conn.close()


def upsert_market(slug, question, category, asset, timeframe, start_date, end_date):
    conn = get_conn()
    existing = conn.execute("SELECT id FROM markets WHERE slug=?", (slug,)).fetchone()
    if existing:
        conn.close()
        return existing["id"]
    cur = conn.execute("""
        INSERT INTO markets (slug, question, category, asset, timeframe, start_date, end_date, first_seen_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, (slug, question, category, asset, timeframe, start_date, end_date, time.time()))
    conn.commit()
    market_id = cur.lastrowid
    conn.close()
    return market_id


def insert_market_snapshot(market_id, yes_price, no_price, volume_24hr, liquidity,
                            best_bid=None, best_ask=None, ask_depth_usd=None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO market_snapshots (market_id, ts, yes_price, no_price, volume_24hr, liquidity,
                                       best_bid, best_ask, ask_depth_usd)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (market_id, time.time(), yes_price, no_price, volume_24hr, liquidity, best_bid, best_ask, ask_depth_usd))
    conn.commit()
    conn.close()


def insert_external_snapshot(asset, spot_price, funding_rate, open_interest):
    conn = get_conn()
    conn.execute("""
        INSERT INTO external_snapshots (asset, ts, spot_price, funding_rate, open_interest)
        VALUES (?,?,?,?,?)
    """, (asset, time.time(), spot_price, funding_rate, open_interest))
    conn.commit()
    conn.close()


def insert_prediction(market_id, market_prob, model_prob, edge, confidence, feature_scores, explanation):
    conn = get_conn()
    conn.execute("""
        INSERT INTO predictions (market_id, ts, market_prob, model_prob, edge, confidence,
                                  feature_scores_json, explanation)
        VALUES (?,?,?,?,?,?,?,?)
    """, (market_id, time.time(), market_prob, model_prob, edge, confidence,
          json.dumps(feature_scores), explanation))
    conn.commit()
    conn.close()


def insert_resolution(market_id, outcome_up):
    conn = get_conn()
    conn.execute("""
        INSERT INTO resolutions (market_id, resolved_at, outcome_up) VALUES (?,?,?)
    """, (market_id, time.time(), int(outcome_up)))
    conn.commit()
    conn.close()


def markets_needing_resolution():
    """Markets whose end_date has passed but have no resolution recorded yet.
    This is what makes historical_base_rate (and future accuracy scoring)
    actually possible — without this, resolutions never gets populated."""
    import time as _time
    conn = get_conn()
    rows = conn.execute("""
        SELECT m.* FROM markets m
        LEFT JOIN resolutions r ON r.market_id = m.id
        WHERE r.id IS NULL AND m.end_date IS NOT NULL AND m.end_date < ?
        ORDER BY m.end_date DESC LIMIT 50
    """, (_time.time(),)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def predictions_for_market(market_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM predictions WHERE market_id=? ORDER BY id DESC", (market_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_backtest_run(category, asset, timeframe, n_markets, accuracy, brier, log_loss_val,
                      sharpe, max_dd, per_feature_accuracy, weights_used):
    conn = get_conn()
    conn.execute("""
        INSERT INTO backtest_runs (run_at, category, asset, timeframe, n_markets, accuracy,
                                    brier_score, log_loss, sharpe_ratio, max_drawdown,
                                    per_feature_accuracy_json, weights_used_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (time.time(), category, asset, timeframe, n_markets, accuracy, brier, log_loss_val,
          sharpe, max_dd, json.dumps(per_feature_accuracy), json.dumps(weights_used)))
    conn.commit()
    conn.close()


def resolved_predictions(limit=1000):
    """Every prediction whose market has since resolved, paired with the
    real outcome. This is the ground truth v0.5 (and the live accuracy
    view) both depend on."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT p.*, r.outcome_up, m.asset, m.timeframe, m.question
        FROM predictions p
        JOIN resolutions r ON r.market_id = p.market_id
        JOIN markets m ON m.id = p.market_id
        ORDER BY p.id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def data_collection_stats():
    conn = get_conn()
    total_predictions = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    total_resolved = conn.execute("SELECT COUNT(*) c FROM resolutions").fetchone()["c"]
    oldest = conn.execute("SELECT MIN(ts) t FROM predictions").fetchone()["t"]
    conn.close()
    import time as _time
    days_collecting = round((_time.time() - oldest) / 86400, 1) if oldest else 0
    return {
        "total_predictions": total_predictions,
        "total_resolved": total_resolved,
        "days_collecting": days_collecting,
    }


def log_feedback_run(n_predictions, old_weights, new_weights, per_feature_accuracy, summary):
    import json
    conn = get_conn()
    conn.execute("""
        INSERT INTO feedback_runs (run_at, n_predictions_analyzed, old_weights_json, new_weights_json,
                                    per_feature_accuracy_json, summary)
        VALUES (?,?,?,?,?,?)
    """, (time.time(), n_predictions, json.dumps(old_weights), json.dumps(new_weights),
          json.dumps(per_feature_accuracy), summary))
    conn.commit()
    conn.close()


def latest_feedback_run():
    conn = get_conn()
    row = conn.execute("SELECT * FROM feedback_runs ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


def latest_backtest_runs(limit=20):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM backtest_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def recent_predictions(limit=50):
    conn = get_conn()
    rows = conn.execute("""
        SELECT p.*, m.question, m.asset, m.timeframe FROM predictions p
        JOIN markets m ON m.id = p.market_id
        ORDER BY p.id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
