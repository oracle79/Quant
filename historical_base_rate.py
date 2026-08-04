"""
features/historical_base_rate.py — "how often has THIS type of market
(same asset, same timeframe) actually resolved Up historically?" Queries
our own resolutions table — this feature literally gets smarter as the
platform accumulates history, independent of any other feature.
"""
from .base import Feature, FeatureResult


class HistoricalBaseRateFeature(Feature):
    name = "historical_base_rate"

    def compute(self, context: dict) -> FeatureResult:
        db = context["db_module"]
        asset = context["asset"]
        timeframe = context["timeframe"]

        conn = db.get_conn()
        rows = conn.execute("""
            SELECT r.outcome_up FROM resolutions r
            JOIN markets m ON m.id = r.market_id
            WHERE m.asset=? AND m.timeframe=?
            ORDER BY r.id DESC LIMIT 500
        """, (asset, timeframe)).fetchall()
        conn.close()

        n = len(rows)
        if n < 20:
            # not enough history yet to say anything — stay neutral, low confidence
            return FeatureResult(
                name=self.name, score=0.5, confidence=0.05,
                explanation=f"Only {n} resolved {asset} {timeframe} markets on record "
                            f"(need 20+) — defaulting to neutral.",
            )
        up_rate = sum(r["outcome_up"] for r in rows) / n
        confidence = min(1.0, n / 200)  # more history -> more confidence, caps at 200 samples
        return FeatureResult(
            name=self.name, score=up_rate, confidence=confidence,
            explanation=f"{asset} {timeframe} markets resolved Up {up_rate*100:.1f}% "
                        f"of the time across the last {n} resolutions.",
        )
