"""
optimize_weights.py — v0.5 from the spec: "every feature must prove that it
improves predictive performance before remaining in the model."

For each resolved prediction, checks whether each feature's own score
(sign: >0.5 = predicted Up) matched the real outcome. Features that
predicted correctly more often than chance get nudged up; features at or
below chance get nudged down. Requires a minimum sample size before
touching anything — with too few resolved predictions, this is more noise
than signal, and changing weights on noise is worse than leaving them alone.
"""
import sys
import os
import json
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import DEFAULT_WEIGHTS, load_weights, save_weights
from database import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("optimize_weights")

MIN_PREDICTIONS_TO_ADAPT = 50
LEARNING_RATE = 0.15
MIN_WEIGHT = 0.02


def analyze_feature_performance(lookback=1000):
    resolved = db.resolved_predictions(limit=lookback)
    stats = {k: {"correct": 0, "n": 0} for k in DEFAULT_WEIGHTS}

    for row in resolved:
        try:
            feature_scores = json.loads(row["feature_scores_json"])
        except Exception:
            continue
        actual_up = bool(row["outcome_up"])
        for feat_name, feat_result in feature_scores.items():
            if feat_name not in stats or feat_result is None:
                continue
            score = feat_result.get("score")
            confidence = feat_result.get("confidence", 0)
            if score is None or confidence < 0.15:
                continue  # feature had nothing meaningful to say on this one
            predicted_up = score >= 0.5
            stats[feat_name]["n"] += 1
            if predicted_up == actual_up:
                stats[feat_name]["correct"] += 1

    result = {}
    for k, s in stats.items():
        result[k] = {"accuracy": (s["correct"] / s["n"]) if s["n"] > 0 else None, "n": s["n"]}
    return result, len(resolved)


def compute_new_weights(old_weights, per_feature_accuracy):
    new_weights = dict(old_weights)
    for k, stat in per_feature_accuracy.items():
        if stat["n"] < 15 or stat["accuracy"] is None:
            continue  # too few samples for THIS feature specifically, leave it alone
        edge = stat["accuracy"] - 0.5
        adjustment = edge * LEARNING_RATE
        new_weights[k] = max(MIN_WEIGHT, old_weights[k] * (1 + adjustment))
    total = sum(new_weights.values()) or 1.0
    return {k: v / total for k, v in new_weights.items()}


RETIREMENT_STATE_PATH = os.environ.get("RETIREMENT_STATE_PATH", "retirement_state.json")
RETIREMENT_THRESHOLD_RUNS = 3  # consecutive weeks at the floor before flagging


def _load_retirement_state():
    if os.path.exists(RETIREMENT_STATE_PATH):
        try:
            with open(RETIREMENT_STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _update_retirement_tracking(new_weights):
    """Tracks how many consecutive optimization runs each feature has spent
    pinned at the weight floor -- a feature stuck there repeatedly isn't
    earning its place (Boris Principle #5: reject weak ideas automatically,
    don't just let them quietly sit at a token weight forever)."""
    state = _load_retirement_state()
    candidates = []
    for feat, w in new_weights.items():
        at_floor = w <= MIN_WEIGHT * 1.05  # small tolerance for float drift
        if at_floor:
            state[feat] = state.get(feat, 0) + 1
        else:
            state[feat] = 0
        if state[feat] >= RETIREMENT_THRESHOLD_RUNS:
            candidates.append(feat)
    with open(RETIREMENT_STATE_PATH, "w") as f:
        json.dump(state, f)
    return candidates


def run_optimization():
    db.init_db()
    old_weights = load_weights()
    per_feature_accuracy, n_resolved = analyze_feature_performance()

    if n_resolved < MIN_PREDICTIONS_TO_ADAPT:
        summary = (f"Only {n_resolved} resolved predictions (need {MIN_PREDICTIONS_TO_ADAPT}+) — "
                   f"not enough evidence yet. Weights unchanged.")
        log.info(summary)
        db.log_feedback_run(n_resolved, old_weights, old_weights, per_feature_accuracy, summary)
        return

    new_weights = compute_new_weights(old_weights, per_feature_accuracy)
    retirement_candidates = _update_retirement_tracking(new_weights)

    lines = [f"Reviewed {n_resolved} resolved predictions."]
    for k in old_weights:
        stat = per_feature_accuracy.get(k, {})
        acc = stat.get("accuracy")
        acc_str = f"{acc*100:.1f}% (n={stat.get('n',0)})" if acc is not None else "insufficient data"
        lines.append(f"  {k}: accuracy {acc_str} | weight {old_weights[k]:.3f} -> {new_weights[k]:.3f}")
    if retirement_candidates:
        lines.append(f"\n⚠️ Retirement candidates (weak for {RETIREMENT_THRESHOLD_RUNS}+ consecutive weeks): "
                      f"{', '.join(retirement_candidates)} — consider removing or redesigning these.")
    summary = "\n".join(lines)
    log.info(summary)

    save_weights(new_weights)
    db.log_feedback_run(n_resolved, old_weights, new_weights, per_feature_accuracy, summary)

    try:
        from telegram import notifier
        notifier.send_message(f"🔧 Weekly weight optimization:\n{summary}")
    except Exception:
        pass


if __name__ == "__main__":
    run_optimization()
